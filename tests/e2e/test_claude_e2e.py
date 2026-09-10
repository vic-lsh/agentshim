"""Claude Code against the real binary."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentshim import CliAgent, OutputSchema, TurnRequest
from tests.e2e.conftest import requires_cli

pytestmark = [pytest.mark.e2e, requires_cli("claude")]


def test_a_single_turn_answers_and_reports_usage() -> None:
    result = CliAgent("claude").run("Reply with exactly: pong")
    assert "pong" in result.text.lower()
    assert result.session_id is not None
    assert result.usage.tokens.input_tokens > 0
    assert result.usage.tokens.cached_input_tokens <= result.usage.tokens.input_tokens


def test_a_second_turn_resumes_the_first() -> None:
    session = CliAgent("claude").start_session()
    session.turn("Remember the word 'juniper'. Reply with 'ok'.")
    assert session.session_id is not None
    second = session.turn("What word did I ask you to remember? Reply with just the word.")
    assert second.resumed is True
    assert "juniper" in second.text.lower()


def test_a_native_output_schema_returns_structured_output(tmp_path: Path) -> None:
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "integer"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    result = (
        CliAgent("claude")
        .start_session(cwd=str(tmp_path))
        .turn(TurnRequest(prompt="What is 2 + 2?", output_schema=OutputSchema(schema=schema, host_dir=tmp_path)))
    )
    assert result.structured_output == {"answer": 4}
