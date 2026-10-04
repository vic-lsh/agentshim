"""Codex against the real binary."""

from __future__ import annotations

from pathlib import Path

import pytest
from agentshim import (
    AgentEventHandler,
    CliAgent,
    OutputSchema,
    SchemaDialect,
    ToolCall,
    TurnRequest,
    dialect_problems,
)
from agentshim.testing import RecordingEventHandler

from tests.e2e.conftest import CODEX_MODEL_VAR, model_from_env, requires_cli

pytestmark = [pytest.mark.e2e, requires_cli("codex")]


def _agent(event_handler: AgentEventHandler | None = None) -> CliAgent:
    return CliAgent("codex", model=model_from_env(CODEX_MODEL_VAR), event_handler=event_handler)


def test_a_single_turn_answers_and_reports_usage(tmp_path: Path) -> None:
    result = _agent().run("Reply with the single word pong.", cwd=str(tmp_path))
    assert "pong" in result.text.lower()
    assert result.session_id is not None
    assert result.usage.tokens.input_tokens > 0
    assert result.usage.tokens.cached_input_tokens <= result.usage.tokens.input_tokens


def test_a_second_turn_resumes_the_first(tmp_path: Path) -> None:
    session = _agent().start_session(cwd=str(tmp_path))
    first = session.turn("Remember the word 'juniper'. Reply with 'ok'.")
    assert session.session_id is not None
    second = session.turn("What word did I ask you to remember? Reply with just the word.")
    assert second.resumed is True
    assert "juniper" in second.text.lower()
    assert first.usage.raw is not None
    assert second.usage.raw is not None
    summed = first.usage.tokens + second.usage.tokens
    for raw_field, normalized_field in (
        ("input_tokens", "input_tokens"),
        ("cached_input_tokens", "cache_read_input_tokens"),
        ("cache_write_input_tokens", "cache_write_input_tokens"),
        ("output_tokens", "output_tokens"),
        ("reasoning_output_tokens", "reasoning_output_tokens"),
    ):
        assert getattr(summed, normalized_field) == second.usage.raw.get(raw_field, 0)


def test_a_native_output_schema_returns_structured_output(tmp_path: Path) -> None:
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "integer"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    result = (
        _agent()
        .start_session(cwd=str(tmp_path))
        .turn(
            TurnRequest(
                prompt="What is 2 + 2?",
                output_schema=OutputSchema(schema=schema, host_dir=tmp_path),
            )
        )
    )
    assert result.structured_output == {"answer": 4}


def test_a_schema_the_strict_check_accepts_is_accepted_live(tmp_path: Path) -> None:
    """Nested, nullable, ``anyOf`` and ``$defs`` forms pass Codex strict mode."""
    schema = {
        "type": "object",
        "properties": {
            "changed": {"type": "boolean"},
            "no_change_reason": {"type": ["string", "null"]},
            "items": {"type": "array", "items": {"$ref": "#/$defs/Item"}},
            "note": {"anyOf": [{"$ref": "#/$defs/Item"}, {"type": "null"}]},
        },
        "required": ["changed", "no_change_reason", "items", "note"],
        "additionalProperties": False,
        "$defs": {
            "Item": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "count": {"type": "integer"}},
                "required": ["name", "count"],
                "additionalProperties": False,
            }
        },
    }
    assert dialect_problems(schema, SchemaDialect.STRICT) == []
    result = (
        _agent()
        .start_session(cwd=str(tmp_path))
        .turn(
            TurnRequest(
                prompt=(
                    "Reply with changed=false, no_change_reason='none needed', "
                    "items=[{name:'a', count:1}], note=null."
                ),
                output_schema=OutputSchema(schema=schema, host_dir=tmp_path),
            )
        )
    )
    assert isinstance(result.structured_output, dict)
    assert result.structured_output["changed"] is False
    assert result.structured_output["items"] == [{"name": "a", "count": 1}]


def test_a_tool_call_is_reported(tmp_path: Path) -> None:
    recorder = RecordingEventHandler()
    result = _agent(event_handler=recorder).run(
        "Run `echo hi` and reply with its output.", cwd=str(tmp_path)
    )
    calls = [event for event in recorder.events if isinstance(event, ToolCall)]
    assert any("echo hi" in str(call.args) for call in calls)
    assert "hi" in result.text.lower()
