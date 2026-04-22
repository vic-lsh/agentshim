"""Regression test for CLICodingAgent._check_cli() stdin handling.

When the CLI probe subprocess inherits the parent's stdin (e.g. a TTY),
the child can receive SIGTTIN/SIGTTOU and become a stopped process,
deadlocking the parent on subprocess.run().wait(). The probe must
therefore always pass stdin=subprocess.DEVNULL.
"""

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from agentshim.claude import ClaudeCodeCodingAgent


@pytest.fixture
def mocked_binary(monkeypatch):
    monkeypatch.setattr(
        "agentshim.cli_agent.shutil.which",
        lambda cmd, path=None: f"/usr/local/bin/{cmd}",
    )


def test_check_cli_probe_passes_stdin_devnull(mocked_binary):
    """_check_cli must pass stdin=DEVNULL so the probe never reads the TTY."""
    completed = MagicMock(returncode=0, stderr="")

    with patch("agentshim.cli_agent.subprocess.run", return_value=completed) as run:
        ClaudeCodeCodingAgent(model="test-model")

    # Find the --help probe call (there may be other subprocess.run calls).
    probe_calls = [c for c in run.call_args_list if c.args and c.args[0][-1] == "--help"]
    assert probe_calls, "expected a --help probe call from _check_cli"
    for call in probe_calls:
        assert call.kwargs.get("stdin") is subprocess.DEVNULL, (
            f"probe subprocess.run must pass stdin=subprocess.DEVNULL, got kwargs={call.kwargs}"
        )
