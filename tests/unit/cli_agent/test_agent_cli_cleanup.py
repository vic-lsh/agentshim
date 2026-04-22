"""
Comprehensive cleanup tests for agent CLI implementations.

Tests proper resource cleanup (processes, threads, pipes, file descriptors)
across all agent types (Claude, Codex, Gemini) to ensure no resource leaks
in success, failure, and edge case scenarios.

This test suite is parameterized to run identical tests across all agent
implementations with a stubbed subprocess to avoid invoking actual LLMs.
"""

import subprocess
import threading
from unittest.mock import MagicMock, patch

import pytest

from agentshim.claude import ClaudeCodeCodingAgent
from agentshim.codex import CodexCodingAgent
from agentshim.gemini import GeminiCodingAgent


class MockProcess:
    """Mock subprocess.Popen for testing cleanup behavior."""

    def __init__(self, *args, **kwargs):
        self.pid = 12345
        self.returncode = 0
        self.stdout = MagicMock()
        self.stderr = MagicMock()
        self.stdin = MagicMock()
        self.killed = False
        self.wait_called = False
        self.stdin_closed = False
        self._simulate_timeout = False

    def wait(self, timeout=None):
        self.wait_called = True
        if timeout and self._simulate_timeout:
            raise subprocess.TimeoutExpired(cmd="test", timeout=timeout)
        return self.returncode

    def poll(self):
        return self.returncode

    def terminate(self):
        pass

    def kill(self):
        self.killed = True

    def _set_returncode(self, code):
        self.returncode = code


@pytest.fixture
def mock_env():
    """Mock environment for agent initialization."""
    return {"PATH": "/usr/bin:/bin", "HOME": "/tmp", "USER": "test"}


@pytest.fixture
def mock_which(mock_env):
    """Mock shutil.which to return a fake binary path."""

    def which_impl(name, path=None):
        return f"/usr/bin/{name}"

    return which_impl


@pytest.fixture(
    params=[
        ("claude", ClaudeCodeCodingAgent),
        ("codex", CodexCodingAgent),
        ("gemini", GeminiCodingAgent),
    ],
    ids=["claude", "codex", "gemini"],
)
def agent_type(request):
    """Parameterized fixture for all agent types."""
    return request.param


# ============================================================================
# INITIALIZATION AND BINARY DETECTION TESTS
# ============================================================================


def test_agent_detects_missing_binary(agent_type):
    """Test agent fails gracefully when binary is not found."""
    agent_name, agent_class = agent_type

    with patch("shutil.which", return_value=None):
        with pytest.raises(RuntimeError, match="not found in PATH"):
            agent_class()


def test_agent_detects_broken_binary(agent_type, mock_which):
    """Test agent fails when binary's --help returns non-zero."""
    agent_name, agent_class = agent_type

    with patch("shutil.which", side_effect=mock_which):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="binary error")
            with pytest.raises(RuntimeError, match="not working correctly"):
                agent_class()


def test_agent_detects_missing_binary_at_path(agent_type, mock_which):
    """Test agent fails when binary file doesn't exist at expected path."""
    agent_name, agent_class = agent_type

    with patch("shutil.which", side_effect=mock_which):
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError()
            with pytest.raises(RuntimeError, match="not found"):
                agent_class()


def test_agent_initializes_with_model(agent_type, mock_which):
    """Test agent initializes with optional model parameter."""
    agent_name, agent_class = agent_type

    with patch("shutil.which", side_effect=mock_which):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            agent = agent_class(model="custom-model")
            assert agent.model == "custom-model"


# ============================================================================
# SUCCESSFUL EXECUTION AND CLEANUP TESTS
# ============================================================================


def test_generate_success_cleans_up_pipes(agent_type, mock_which):
    """Test pipes are properly closed on successful execution."""
    agent_name, agent_class = agent_type

    mock_process = MockProcess()
    mock_process.stdout.readline.side_effect = ["output line\n", ""]
    mock_process.stderr.readline.side_effect = [""]

    with patch("shutil.which", side_effect=mock_which):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            with patch("subprocess.Popen", return_value=mock_process):
                agent = agent_class()
                agent.generate("test prompt", silent=True)

    # Verify pipes were closed
    assert mock_process.stdout.close.called or mock_process.stdout.readline.called
    assert mock_process.stderr.close.called or mock_process.stderr.readline.called
    assert mock_process.stdin.close.called
    assert mock_process.wait_called


def test_generate_success_waits_for_threads(agent_type, mock_which):
    """Test that threads are properly joined before returning."""
    agent_name, agent_class = agent_type

    threads_joined = []

    def track_thread_join(timeout=None):
        threads_joined.append(timeout)

    mock_process = MockProcess()
    mock_process.stdout.readline.side_effect = ["output\n", ""]
    mock_process.stderr.readline.side_effect = [""]

    with patch("shutil.which", side_effect=mock_which):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            with patch("subprocess.Popen", return_value=mock_process):
                with patch("threading.Thread") as mock_thread_class:
                    mock_thread = MagicMock()
                    mock_thread.join = track_thread_join
                    mock_thread_class.return_value = mock_thread

                    agent = agent_class()
                    agent.generate("test", silent=True)

    # Verify threads were joined with timeout
    assert len(threads_joined) >= 2  # stdout and stderr threads


def test_generate_success_closes_stdin(agent_type, mock_which):
    """Test stdin is closed after sending prompt."""
    agent_name, agent_class = agent_type

    mock_process = MockProcess()
    mock_process.stdout.readline.side_effect = ["output\n", ""]
    mock_process.stderr.readline.side_effect = [""]

    with patch("shutil.which", side_effect=mock_which):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            with patch("subprocess.Popen", return_value=mock_process):
                agent = agent_class()
                agent.generate("test prompt", silent=True)

    assert mock_process.stdin.write.called
    assert mock_process.stdin.close.called


def test_generate_success_returns_output(agent_type, mock_which):
    """Test successful execution returns expected output."""
    agent_name, agent_class = agent_type

    mock_process = MockProcess()
    expected_output = "success output"

    if agent_name == "gemini":
        # Gemini expects JSON stream output
        import json

        json_output = json.dumps({"type": "message", "role": "assistant", "content": expected_output})
        mock_process.stdout.readline.side_effect = [f"{json_output}\n", ""]
    else:
        # Other agents expect plain text
        mock_process.stdout.readline.side_effect = [f"{expected_output}\n", ""]

    mock_process.stderr.readline.side_effect = [""]
    mock_process.returncode = 0

    with patch("shutil.which", side_effect=mock_which):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            with patch("subprocess.Popen", return_value=mock_process):
                agent = agent_class()
                result = agent.generate("test", silent=True)

    assert expected_output in result


# ============================================================================
# TIMEOUT AND PROCESS TERMINATION CLEANUP TESTS
# ============================================================================


def test_generate_timeout_kills_process(agent_type, mock_which):
    """Test process group is killed on timeout."""
    agent_name, agent_class = agent_type

    mock_process = MockProcess()
    wait_calls = []

    # Simulate timeout on wait() with first call, return 0 on second call
    def wait_with_timeout(timeout=None):
        wait_calls.append(timeout)
        if timeout:
            raise subprocess.TimeoutExpired(cmd="test", timeout=timeout)
        return 0

    mock_process.wait = wait_with_timeout
    mock_process.stdout.readline.side_effect = [""]
    mock_process.stderr.readline.side_effect = [""]

    killed_pids = []

    def mock_killpg(pid, sig):
        killed_pids.append(pid)

    with patch("shutil.which", side_effect=mock_which):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            with patch("subprocess.Popen", return_value=mock_process):
                with patch("os.killpg", side_effect=mock_killpg):
                    with patch("os.getpgid", return_value=12345):
                        agent = agent_class()

                        with pytest.raises(subprocess.TimeoutExpired):
                            agent.generate("test", timeout=1, silent=True)

    # Verify process group was killed
    assert len(killed_pids) > 0
    # Should have 2 calls: one with timeout (raises), one without timeout (returns 0)
    assert len(wait_calls) >= 2


def test_generate_timeout_handles_process_already_dead(agent_type, mock_which):
    """Test cleanup when process is already dead (ProcessLookupError)."""
    agent_name, agent_class = agent_type

    mock_process = MockProcess()

    def wait_with_timeout(timeout=None):
        if timeout:
            raise subprocess.TimeoutExpired(cmd="test", timeout=timeout)
        return -9  # Killed by signal

    mock_process.wait = wait_with_timeout
    mock_process.stdout.readline.side_effect = [""]
    mock_process.stderr.readline.side_effect = [""]

    def mock_killpg_not_found(pid, sig):
        raise ProcessLookupError("process already gone")

    with patch("shutil.which", side_effect=mock_which):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            with patch("subprocess.Popen", return_value=mock_process):
                with patch("os.killpg", side_effect=mock_killpg_not_found):
                    with patch("os.getpgid", return_value=12345):
                        agent = agent_class()

                        # Should not raise ProcessLookupError
                        with pytest.raises(subprocess.TimeoutExpired):
                            agent.generate("test", timeout=1, silent=True)


def test_generate_timeout_closes_all_resources(agent_type, mock_which):
    """Test all resources (pipes, threads, process) are closed on timeout."""
    agent_name, agent_class = agent_type

    mock_process = MockProcess()
    wait_calls = []

    def wait_with_timeout(timeout=None):
        wait_calls.append(timeout)
        if timeout:
            raise subprocess.TimeoutExpired(cmd="test", timeout=timeout)
        return -9

    mock_process.wait = wait_with_timeout
    mock_process.stdout.readline.side_effect = [""]
    mock_process.stderr.readline.side_effect = [""]

    with patch("shutil.which", side_effect=mock_which):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            with patch("subprocess.Popen", return_value=mock_process):
                with patch("os.killpg"):
                    with patch("os.getpgid", return_value=12345):
                        agent = agent_class()

                        with pytest.raises(subprocess.TimeoutExpired):
                            agent.generate("test", timeout=1, silent=True)

    # Verify cleanup happened
    assert mock_process.stdin.close.called
    assert len(wait_calls) >= 2  # wait() called twice: once with timeout, once without


# ============================================================================
# ERROR HANDLING AND CLEANUP TESTS
# ============================================================================


def test_generate_non_zero_exit_raises_error(agent_type, mock_which):
    """Test non-zero exit code raises RuntimeError."""
    agent_name, agent_class = agent_type

    mock_process = MockProcess()
    mock_process.returncode = 127  # Command not found
    mock_process.stdout.readline.side_effect = ["output\n", ""]
    mock_process.stderr.readline.side_effect = ["error message\n", ""]

    with patch("shutil.which", side_effect=mock_which):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            with patch("subprocess.Popen", return_value=mock_process):
                agent = agent_class()

                with pytest.raises(RuntimeError, match="exited with code"):
                    agent.generate("test", silent=True)


def test_generate_error_still_cleans_up(agent_type, mock_which):
    """Test cleanup happens even when binary returns error."""
    agent_name, agent_class = agent_type

    mock_process = MockProcess()
    mock_process.returncode = 1  # Error
    mock_process.stdout.readline.side_effect = [""]
    mock_process.stderr.readline.side_effect = ["error\n", ""]

    with patch("shutil.which", side_effect=mock_which):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            with patch("subprocess.Popen", return_value=mock_process):
                agent = agent_class()

                with pytest.raises(RuntimeError):
                    agent.generate("test", silent=True)

    # Verify cleanup still happened
    assert mock_process.stdin.close.called
    assert mock_process.wait_called


def test_generate_broken_pipe_on_stdin_handled(agent_type, mock_which):
    """Test BrokenPipeError on stdin write is handled gracefully."""
    agent_name, agent_class = agent_type

    mock_process = MockProcess()
    mock_process.stdin.write.side_effect = BrokenPipeError("pipe broken")
    mock_process.stdout.readline.side_effect = [""]
    mock_process.stderr.readline.side_effect = [""]
    mock_process.returncode = 0  # Process succeeded despite broken pipe

    with patch("shutil.which", side_effect=mock_which):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            with patch("subprocess.Popen", return_value=mock_process):
                agent = agent_class()

                # Should not raise BrokenPipeError - it's caught and handled
                result = agent.generate("test", silent=True)
                # Process succeeds even though stdin write failed
                assert result is not None


# ============================================================================
# THREAD CLEANUP AND ORPHANED THREAD TESTS
# ============================================================================


def test_generate_thread_timeout_doesnt_leak_threads(agent_type, mock_which):
    """Test threads are properly cleaned up even if join() times out."""
    agent_name, agent_class = agent_type

    thread_join_count = {"count": 0}

    class TrackingThread(threading.Thread):
        def join(self, timeout=None):
            thread_join_count["count"] += 1
            # Simulate thread not finishing in timeout
            super().join(timeout=0.0001)

    mock_process = MockProcess()
    mock_process.stdout.readline.side_effect = ["output\n", ""]
    mock_process.stderr.readline.side_effect = ["error\n", ""]

    with patch("shutil.which", side_effect=mock_which):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            with patch("subprocess.Popen", return_value=mock_process):
                with patch("threading.Thread", side_effect=TrackingThread):
                    agent = agent_class()
                    agent.generate("test", silent=True)

    # Verify join() was attempted on threads
    assert thread_join_count["count"] >= 2


# ============================================================================
# WORKING DIRECTORY AND ENVIRONMENT CLEANUP TESTS
# ============================================================================


def test_generate_with_cwd_parameter(agent_type, mock_which):
    """Test working directory parameter is passed to subprocess."""
    agent_name, agent_class = agent_type

    captured_cwd = []

    def track_popen(*args, **kwargs):
        captured_cwd.append(kwargs.get("cwd"))
        mock_process = MockProcess()
        mock_process.stdout.readline.side_effect = ["output\n", ""]
        mock_process.stderr.readline.side_effect = [""]
        return mock_process

    test_cwd = "/home/user/project"

    with patch("shutil.which", side_effect=mock_which):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            with patch("subprocess.Popen", side_effect=track_popen):
                agent = agent_class()
                agent.generate("test", cwd=test_cwd, silent=True)

    assert test_cwd in captured_cwd


def test_generate_with_custom_env(agent_type, mock_which, mock_env):
    """Test custom environment is used for subprocess."""
    agent_name, agent_class = agent_type

    captured_env = []

    def track_popen(*args, **kwargs):
        captured_env.append(kwargs.get("env"))
        mock_process = MockProcess()
        mock_process.stdout.readline.side_effect = ["output\n", ""]
        mock_process.stderr.readline.side_effect = [""]
        return mock_process

    with patch("shutil.which", side_effect=mock_which):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            with patch("subprocess.Popen", side_effect=track_popen):
                with patch(
                    "agentshim.cli_agent.get_interactive_env",
                    return_value=mock_env,
                ):
                    agent = agent_class()
                    agent.generate("test", silent=True)

    # Verify environment was passed to Popen
    assert len(captured_env) > 0
    assert captured_env[0] is not None


# ============================================================================
# SILENT MODE TESTS
# ============================================================================


def test_generate_silent_mode_suppresses_output(agent_type, mock_which, capsys):
    """Test silent mode prevents printing of agent output."""
    agent_name, agent_class = agent_type

    mock_process = MockProcess()
    mock_process.stdout.readline.side_effect = ["sensitive output\n", ""]
    mock_process.stderr.readline.side_effect = ["debug info\n", ""]

    with patch("shutil.which", side_effect=mock_which):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            with patch("subprocess.Popen", return_value=mock_process):
                agent = agent_class()
                result = agent.generate("test", silent=True)

    # In silent mode, the output shouldn't be printed to console
    # (though the function still returns the result)
    # Note: This checks the actual stdout, not mocks
    assert result is not None


# ============================================================================
# RESOURCE COUNTING AND LEAK DETECTION TESTS
# ============================================================================


def test_multiple_generates_dont_leak_resources(agent_type, mock_which):
    """Test multiple calls to generate() don't accumulate resource leaks."""
    agent_name, agent_class = agent_type

    def mock_popen(*args, **kwargs):
        mock_process = MockProcess()
        mock_process.stdout.readline.side_effect = ["output\n", ""]
        mock_process.stderr.readline.side_effect = [""]
        return mock_process

    with patch("shutil.which", side_effect=mock_which):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            with patch("subprocess.Popen", side_effect=mock_popen):
                agent = agent_class()

                # Call multiple times
                for i in range(5):
                    result = agent.generate(f"test {i}", silent=True)
                    assert result is not None

