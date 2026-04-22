"""Fixtures for LLM tests."""

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_llm_subprocess(monkeypatch):
    """Mock subprocess operations for LLM agent tests."""
    mock_popen = MagicMock()
    mock_process = MagicMock()
    mock_process.returncode = 0
    mock_process.communicate.return_value = (b"", b"")
    mock_process.poll.return_value = 0
    mock_process.stdout = MagicMock()
    mock_process.stdout.readline = MagicMock(return_value="")
    mock_process.stderr = MagicMock()
    mock_process.stderr.readline = MagicMock(side_effect=lambda: "")
    mock_popen.return_value = mock_process

    mock_which = MagicMock()
    mock_which.return_value = "/usr/bin/agent"

    monkeypatch.setattr("subprocess.Popen", mock_popen)
    monkeypatch.setattr("shutil.which", mock_which)

    return mock_popen, mock_which
