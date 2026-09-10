"""Codex against the real binary."""

from __future__ import annotations

from pathlib import Path

import pytest
from agentshim import CliAgent, OutputSchema, ToolCall, TurnRequest
from agentshim.testing import RecordingEventHandler

from tests.e2e.conftest import requires_cli

pytestmark = [pytest.mark.e2e, requires_cli("codex")]


def test_a_single_turn_answers_and_reports_usage(tmp_path: Path) -> None:
    result = CliAgent("codex").run("Reply with the single word pong.", cwd=str(tmp_path))
    assert "pong" in result.text.lower()
    assert result.session_id is not None
    assert result.usage.tokens.input_tokens > 0
    assert result.usage.tokens.cached_input_tokens <= result.usage.tokens.input_tokens


def test_a_second_turn_resumes_the_first(tmp_path: Path) -> None:
    session = CliAgent("codex").start_session(cwd=str(tmp_path))
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
        CliAgent("codex")
        .start_session(cwd=str(tmp_path))
        .turn(
            TurnRequest(
                prompt="What is 2 + 2?",
                output_schema=OutputSchema(schema=schema, host_dir=tmp_path),
            )
        )
    )
    assert result.structured_output == {"answer": 4}


def test_a_tool_call_is_reported(tmp_path: Path) -> None:
    recorder = RecordingEventHandler()
    result = CliAgent("codex", event_handler=recorder).run(
        "Run `echo hi` and reply with its output.", cwd=str(tmp_path)
    )
    calls = [event for event in recorder.events if isinstance(event, ToolCall)]
    assert any("echo hi" in str(call.args) for call in calls)
    assert "hi" in result.text.lower()
