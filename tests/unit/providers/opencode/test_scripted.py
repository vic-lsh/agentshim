"""The scripted opencode stream round-trips through the real parser."""

from __future__ import annotations

import json

import pytest
from agentshim import (
    AssistantText,
    CliAgent,
    SessionStarted,
    TokenUsage,
    ToolCall,
    ToolResult,
    UsageReport,
)
from agentshim.testing import FakeExecutor, RecordingEventHandler, scripted_turn

_ENV = {"PATH": "/usr/bin"}


def _agent(executor: FakeExecutor, recorder: RecordingEventHandler | None = None) -> CliAgent:
    return CliAgent("opencode", executor=executor, env=dict(_ENV), event_handler=recorder)


class TestScriptedTurn:
    def test_it_round_trips_through_the_real_parser(self) -> None:
        recorder = RecordingEventHandler()
        executor = FakeExecutor(scripted_turn("opencode", text="pong", session_id="ses_1"))

        result = _agent(executor, recorder).run("ping")

        assert result.text == "pong"
        assert result.session_id == "ses_1"
        assert SessionStarted("ses_1") in recorder.events
        assert AssistantText("pong") in recorder.events
        assert any(isinstance(event, UsageReport) for event in recorder.events)

    def test_the_prompt_is_written_to_stdin_not_argv(self) -> None:
        executor = FakeExecutor(scripted_turn("opencode", text="ok"))
        _agent(executor).run("deploy the app")
        request = executor.requests[0]
        assert request.stdin == "deploy the app"
        assert all("deploy the app" not in arg for arg in request.argv)

    def test_a_second_turn_resumes_the_first(self) -> None:
        executor = FakeExecutor(
            [
                scripted_turn("opencode", text="ok", session_id="ses_1"),
                scripted_turn("opencode", text="juniper", session_id="ses_1"),
            ]
        )
        session = _agent(executor).start_session()
        session.turn("remember juniper")
        second = session.turn("what was the word?")
        assert second.resumed is True
        assert executor.requests[1].argv[1:4] == ["run", "--session", "ses_1"]

    def test_every_scripted_line_is_a_json_object(self) -> None:
        run = scripted_turn(
            "opencode", text="hi", session_id="ses_1", tool_calls=[("read", {"path": "/x"}, "out")]
        )
        for line in run.stdout:
            assert line.endswith("\n")
            assert isinstance(json.loads(line), dict)

    def test_tool_calls_become_paired_events(self) -> None:
        recorder = RecordingEventHandler()
        run = scripted_turn(
            "opencode", text="done", tool_calls=[("bash", {"command": "ls"}, "file.txt")]
        )
        _agent(FakeExecutor(run), recorder).run("go")

        calls = [event for event in recorder.events if isinstance(event, ToolCall)]
        results = [event for event in recorder.events if isinstance(event, ToolResult)]
        assert [call.tool for call in calls] == ["bash"]
        assert calls[0].args == {"command": "ls"}
        assert [result.tool for result in results] == ["bash"]
        assert results[0].stdout == "file.txt"

    def test_usage_round_trips_exactly(self) -> None:
        usage = TokenUsage(
            input_tokens=150,
            output_tokens=40,
            cached_input_tokens=50,
            cache_write_input_tokens=10,
            reasoning_output_tokens=10,
            turns=1,
        )
        result = _agent(FakeExecutor(scripted_turn("opencode", text="done", usage=usage))).run("go")
        assert result.usage.tokens == usage
        assert result.usage.tokens.cached_input_tokens <= result.usage.tokens.input_tokens

    def test_a_nonzero_return_code_is_scripted(self) -> None:
        assert scripted_turn("opencode", text="done", returncode=2).returncode == 2

    def test_structured_output_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="no native output schema"):
            scripted_turn("opencode", text="x", structured_output={"a": 1})
