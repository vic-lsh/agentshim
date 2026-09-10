"""The scripted Codex stream round-trips through the real parser."""

from __future__ import annotations

import json
from pathlib import Path

from agentshim import (
    AssistantText,
    CliAgent,
    Lifecycle,
    OutputSchema,
    SessionStarted,
    TokenUsage,
    ToolCall,
    ToolResult,
    TurnRequest,
    UsageReport,
)
from agentshim.testing import FakeExecutor, RecordingEventHandler, scripted_turn

_ENV = {"PATH": "/usr/bin"}
_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "integer"}},
    "required": ["answer"],
    "additionalProperties": False,
}


def _agent(executor: FakeExecutor, recorder: RecordingEventHandler | None = None) -> CliAgent:
    return CliAgent("codex", executor=executor, env=dict(_ENV), event_handler=recorder)


def test_every_scripted_line_is_a_json_object() -> None:
    run = scripted_turn(
        "codex", text="hi", session_id="t-1", tool_calls=[("execute", {"command": "ls"}, "out")]
    )
    for line in run.stdout:
        assert line.endswith("\n")
        assert isinstance(json.loads(line), dict)


def test_a_scripted_turn_round_trips_through_the_real_parser() -> None:
    recorder = RecordingEventHandler()
    executor = FakeExecutor(scripted_turn("codex", text="pong", session_id="t-1"))

    result = _agent(executor, recorder).run("ping")

    assert result.text == "pong"
    assert result.session_id == "t-1"
    assert SessionStarted("t-1") in recorder.events
    assert AssistantText("pong") in recorder.events
    assert Lifecycle("turn_started", "") in recorder.events
    assert any(isinstance(event, UsageReport) for event in recorder.events)


def test_tool_calls_become_paired_events() -> None:
    recorder = RecordingEventHandler()
    executor = FakeExecutor(
        scripted_turn("codex", text="done", tool_calls=[("execute", {"command": "ls"}, "file.txt")])
    )

    _agent(executor, recorder).run("go")

    calls = [event for event in recorder.events if isinstance(event, ToolCall)]
    results = [event for event in recorder.events if isinstance(event, ToolResult)]
    assert [call.args for call in calls] == [{"command": "ls"}]
    assert [(result.tool, result.stdout) for result in results] == [("execute", "file.txt")]


def test_usage_round_trips_exactly() -> None:
    usage = TokenUsage(input_tokens=1200, output_tokens=150, cached_input_tokens=800, turns=1)
    executor = FakeExecutor(scripted_turn("codex", text="done", usage=usage))
    assert _agent(executor).run("go").usage.tokens == usage


def test_structured_output_round_trips_on_a_schema_turn(tmp_path: Path) -> None:
    payload = {"answer": 4}
    executor = FakeExecutor(scripted_turn("codex", structured_output=payload))
    request = TurnRequest(
        prompt="what is 2 + 2?",
        output_schema=OutputSchema(schema=_SCHEMA, host_dir=tmp_path),
    )

    result = _agent(executor).start_session(cwd=str(tmp_path)).turn(request)

    assert result.structured_output == payload
    assert "--output-schema" in executor.requests[0].argv


def test_a_nonzero_return_code_is_scripted() -> None:
    assert scripted_turn("codex", text="done", returncode=2).returncode == 2
