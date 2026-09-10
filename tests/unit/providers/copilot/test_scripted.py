"""The Copilot test-double stream round-trips through the real parser."""

from __future__ import annotations

import json

from agentshim import CliAgent, SessionStarted, TokenUsage, ToolCall, ToolResult, UsageReport
from agentshim.core.events import AgentEvent
from agentshim.providers.copilot import CopilotStreamParser, scripted_lines
from agentshim.testing import FakeExecutor, RecordingEventHandler, scripted_turn

_ENV = {"PATH": "/usr/bin"}


def _agent(executor: FakeExecutor, recorder: RecordingEventHandler | None = None) -> CliAgent:
    return CliAgent("copilot", executor=executor, env=dict(_ENV), event_handler=recorder)


def _replay(lines: list[str]) -> tuple[list[AgentEvent], CopilotStreamParser]:
    events: list[AgentEvent] = []
    parser = CopilotStreamParser(events.append)
    for line in lines:
        parser.feed_stdout(line)
    return events, parser


class TestScriptedLines:
    def test_every_line_is_a_json_object(self) -> None:
        lines = scripted_lines(
            text="hi", session_id="s1", tool_calls=[("shell", {"cmd": "ls"}, "out")]
        )
        for line in lines:
            assert line.endswith("\n")
            assert isinstance(json.loads(line), dict)

    def test_the_frames_are_the_ones_copilot_prints(self) -> None:
        lines = scripted_lines(
            text="hi", session_id="s1", tool_calls=[("shell", {"cmd": "ls"}, "out")]
        )
        kinds = [json.loads(line)["type"] for line in lines]
        assert kinds == [
            "tool.execution_start",
            "tool.execution_complete",
            "assistant.message",
            "assistant.turn_end",
            "result",
        ]

    def test_a_usage_frame_appears_only_when_usage_was_asked_for(self) -> None:
        with_usage = [
            json.loads(line)["type"]
            for line in scripted_lines(text="hi", usage=TokenUsage(turns=1))
        ]
        assert "assistant.usage" in with_usage
        assert "assistant.usage" not in [
            json.loads(line)["type"] for line in scripted_lines(text="hi")
        ]

    def test_structured_output_is_ignored(self) -> None:
        """Copilot has no output-schema mode, so a scripted turn cannot fake one."""
        lines = scripted_lines(text="hi", structured_output={"a": 3})
        assert all("structured" not in line for line in lines)
        _, parser = _replay(lines)
        assert parser.finish().structured_output is None


class TestRoundTrip:
    def test_text_and_session_id_survive(self) -> None:
        events, parser = _replay(scripted_lines(text="pong", session_id="s1"))
        parsed = parser.finish()
        assert parsed.text == "pong"
        assert parsed.session_id == "s1"
        assert SessionStarted("s1") in events

    def test_usage_round_trips_exactly(self) -> None:
        usage = TokenUsage(
            input_tokens=150,
            output_tokens=40,
            cached_input_tokens=50,
            cache_write_input_tokens=30,
            reasoning_output_tokens=7,
            turns=4,
        )
        _, parser = _replay(scripted_lines(text="done", usage=usage))
        assert parser.finish().usage.tokens == usage

    def test_a_turn_without_usage_counts_one_turn(self) -> None:
        _, parser = _replay(scripted_lines(text="done"))
        tokens = parser.finish().usage.tokens
        assert tokens.turns == 1
        assert tokens.input_tokens == 0
        assert tokens.output_tokens == 0

    def test_tool_calls_become_paired_events(self) -> None:
        events, _ = _replay(
            scripted_lines(text="done", tool_calls=[("shell", {"cmd": "ls"}, "file.txt")])
        )
        calls = [event for event in events if isinstance(event, ToolCall)]
        results = [event for event in events if isinstance(event, ToolResult)]
        assert [call.tool for call in calls] == ["shell"]
        assert calls[0].args == {"cmd": "ls"}
        assert [result.tool for result in results] == ["shell"]
        assert results[0].stdout == "file.txt"


class TestScriptedTurn:
    def test_a_scripted_turn_drives_a_whole_agent(self) -> None:
        recorder = RecordingEventHandler()
        executor = FakeExecutor(scripted_turn("copilot", text="pong", session_id="s1"))

        result = _agent(executor, recorder).run("ping")

        assert result.text == "pong"
        assert result.session_id == "s1"
        assert result.structured_output is None
        assert any(isinstance(event, UsageReport) for event in recorder.events)

    def test_a_second_turn_resumes_the_first(self) -> None:
        executor = FakeExecutor(
            [
                scripted_turn("copilot", text="ok", session_id="s1"),
                scripted_turn("copilot", text="still here", session_id="s1"),
            ]
        )
        session = _agent(executor).start_session()
        session.turn("first")
        second = session.turn("second")

        assert second.resumed is True
        assert "--resume" in executor.requests[1].argv
        assert "--resume" not in executor.requests[0].argv

    def test_the_prompt_goes_on_stdin(self) -> None:
        executor = FakeExecutor(scripted_turn("copilot", text="ok"))
        _agent(executor).run("deploy the app")
        assert executor.requests[0].stdin == "deploy the app"
        assert all("deploy the app" not in arg for arg in executor.requests[0].argv)

    def test_a_nonzero_return_code_is_scripted(self) -> None:
        assert scripted_turn("copilot", text="done", returncode=2).returncode == 2
