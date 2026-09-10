"""Copilot CLI against the real binary."""

from __future__ import annotations

import pytest
from agentshim import CliAgent, StdioMcpServer, TurnRequest

from tests.e2e.conftest import requires_cli

pytestmark = [pytest.mark.e2e, requires_cli("copilot")]


def test_a_single_turn_answers_and_reports_a_session() -> None:
    result = CliAgent("copilot").run("Reply with exactly: pong")
    assert "pong" in result.text.lower()
    assert result.session_id is not None
    assert result.exit_code == 0
    assert result.usage.tokens.cached_input_tokens <= result.usage.tokens.input_tokens


def test_a_second_turn_resumes_the_first() -> None:
    session = CliAgent("copilot").start_session()
    session.turn("Remember the word 'juniper'. Reply with 'ok'.")
    assert session.session_id is not None
    second = session.turn("What word did I ask you to remember? Reply with just the word.")
    assert second.resumed is True
    assert "juniper" in second.text.lower()


def test_an_mcp_server_is_installed_from_the_command_line() -> None:
    """The flag has to parse; the turn only has to survive it."""
    server = StdioMcpServer(name="probe", command="cat", args=[])
    result = CliAgent("copilot").run(
        TurnRequest(prompt="Reply with exactly: ok", mcp_servers=[server])
    )
    assert result.exit_code == 0
