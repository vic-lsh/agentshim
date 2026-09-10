"""Recorded Copilot CLI runs, replayed through the parser.

The fixtures under ``tests/fixtures/copilot`` are real ``--output-format
json`` stdout captured from the CLI. They are what keeps the parser honest
about the frames Copilot actually prints, as opposed to the ones its docs
describe.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from agentshim import AssistantText, CliAgent, RawOutput, ToolCall, ToolResult, UsageReport
from agentshim.core.events import AgentEvent
from agentshim.core.provider import ParsedTurn
from agentshim.providers.copilot import CopilotStreamParser
from agentshim.testing import FakeExecutor, FakeRun, RecordingEventHandler

FIXTURE_DIR = Path(__file__).resolve().parents[3] / "fixtures" / "copilot"
RESUMED_SESSION_ID = "33333333-3333-4333-8333-333333333333"
CHECKPOINT_SESSION_ID = "44444444-4444-4444-8444-444444444444"


def fixture_lines(name: str) -> list[str]:
    """Return one recording's stdout, one entry per line."""
    return FIXTURE_DIR.joinpath(name).read_text().splitlines(keepends=True)


def replay(name: str) -> tuple[ParsedTurn, list[AgentEvent]]:
    """Feed one recording to a fresh parser."""
    events: list[AgentEvent] = []
    parser = CopilotStreamParser(events.append)
    for line in fixture_lines(name):
        parser.feed_stdout(line)
    return parser.finish(), events


@pytest.mark.parametrize(
    "name",
    [
        "session_turn_1.jsonl",
        "session_turn_2_resumed.jsonl",
        "streaming_dedup.jsonl",
        "tool_and_usage.jsonl",
        "usage_checkpoint_1_0_83.jsonl",
    ],
)
def test_every_recording_parses_without_raw_output(name: str) -> None:
    """No recorded line is unparseable, and none is mistaken for prose."""
    parsed, events = replay(name)
    assert parsed.error is None
    assert [event for event in events if isinstance(event, RawOutput)] == []


class TestRecordedSession:
    def test_the_first_turn_answers_and_reports_usage(self) -> None:
        parsed, _ = replay("session_turn_1.jsonl")
        assert parsed.text == "STORED"
        assert parsed.session_id == RESUMED_SESSION_ID
        assert parsed.usage.provider == "copilot"
        assert parsed.usage.tokens.output_tokens == 5
        assert parsed.usage.tokens.turns == 1

    def test_the_resumed_turn_keeps_the_same_session_id(self) -> None:
        parsed, _ = replay("session_turn_2_resumed.jsonl")
        assert parsed.text == "CEDAR"
        assert parsed.session_id == RESUMED_SESSION_ID
        assert parsed.usage.tokens.output_tokens == 4

    def test_replaying_both_turns_resumes_the_second(self) -> None:
        executor = FakeExecutor(
            [
                FakeRun(stdout=fixture_lines("session_turn_1.jsonl")),
                FakeRun(stdout=fixture_lines("session_turn_2_resumed.jsonl")),
            ]
        )
        session = CliAgent("copilot", executor=executor, env={"PATH": "/usr/bin"}).start_session()

        first = session.turn("Remember the word CEDAR.")
        second = session.turn("What word did I ask you to remember?")

        assert (first.text, second.text) == ("STORED", "CEDAR")
        assert second.resumed is True
        assert session.session_id == RESUMED_SESSION_ID
        assert "--resume" not in executor.requests[0].argv
        argv = list(executor.requests[1].argv)
        assert argv[argv.index("--resume") + 1] == RESUMED_SESSION_ID


class TestRecordedStreaming:
    def test_a_streamed_message_is_not_duplicated(self) -> None:
        parsed, events = replay("streaming_dedup.jsonl")
        assert parsed.text == "Hello"
        assert [event.text for event in events if isinstance(event, AssistantText)] == ["Hel", "lo"]

    def test_the_message_output_count_is_the_usage_fallback(self) -> None:
        parsed, _ = replay("streaming_dedup.jsonl")
        assert parsed.usage.tokens.output_tokens == 5
        assert parsed.usage.tokens.turns == 1
        assert parsed.session_id == "stream-session"


class TestRecordedToolsAndUsage:
    def test_a_usage_frame_beats_the_message_fallback(self) -> None:
        parsed, _ = replay("tool_and_usage.jsonl")
        assert parsed.text == "done"
        assert parsed.usage.tokens.input_tokens == 115
        assert parsed.usage.tokens.output_tokens == 24
        assert parsed.usage.tokens.cached_input_tokens == 15
        assert parsed.usage.tokens.turns == 1
        assert parsed.usage.tokens.cached_input_tokens <= parsed.usage.tokens.input_tokens

    def test_the_tool_call_and_its_result_are_paired(self) -> None:
        _, events = replay("tool_and_usage.jsonl")
        call = next(event for event in events if isinstance(event, ToolCall))
        result = next(event for event in events if isinstance(event, ToolResult))
        assert (call.tool, call.args) == ("shell", {"command": "printf ok"})
        assert result.tool == "shell"
        assert result.stdout == "ok\nfull"
        assert result.exit_code == 0
        assert result.stderr == ""

    def test_a_whole_turn_through_the_agent_matches_the_recording(self) -> None:
        recorder = RecordingEventHandler()
        executor = FakeExecutor(FakeRun(stdout=fixture_lines("tool_and_usage.jsonl")))
        result = CliAgent(
            "copilot",
            executor=executor,
            env={"PATH": "/usr/bin"},
            event_handler=recorder,
        ).run("run printf ok")

        assert result.text == "done"
        assert result.session_id == "fixture-session"
        assert result.cost_usd is None
        assert result.usage.tokens.input_tokens == 115
        assert {"SessionStarted", "AssistantText", "ToolCall", "ToolResult", "UsageReport"} <= {
            type(event).__name__ for event in recorder.events
        }


class TestRecordedUsageCheckpoint:
    """Copilot CLI 1.0.83 prints no billed token counts.

    The recording is a whole real run. It carries no ``assistant.usage``
    frame and no ``outputTokens``, only a ``session.usage_checkpoint`` whose
    numbers are billing units and prompt-cache diagnostics. Reading those
    into ``TokenUsage`` would report something that is not the turn's usage,
    so the parser leaves them alone; these tests pin that until the CLI
    prints real counts.
    """

    def test_the_run_still_yields_its_text_and_session_id(self) -> None:
        parsed, _ = replay("usage_checkpoint_1_0_83.jsonl")
        assert parsed.text == "pong"
        assert parsed.session_id == CHECKPOINT_SESSION_ID
        assert parsed.error is None

    def test_the_stream_carries_no_usage_frame(self) -> None:
        kinds = {
            json.loads(line)["type"] for line in fixture_lines("usage_checkpoint_1_0_83.jsonl")
        }
        assert "session.usage_checkpoint" in kinds
        assert "assistant.usage" not in kinds

    def test_no_token_counts_are_reported(self) -> None:
        parsed, _ = replay("usage_checkpoint_1_0_83.jsonl")
        tokens = parsed.usage.tokens
        assert tokens.input_tokens == 0
        assert tokens.output_tokens == 0
        assert tokens.cached_input_tokens <= tokens.input_tokens
        assert tokens.turns == 1

    def test_the_streamed_message_is_not_duplicated(self) -> None:
        _, events = replay("usage_checkpoint_1_0_83.jsonl")
        assert [event.text for event in events if isinstance(event, AssistantText)] == ["pong"]

    def test_a_usage_report_is_still_emitted_for_the_turn(self) -> None:
        _, events = replay("usage_checkpoint_1_0_83.jsonl")
        assert [event for event in events if isinstance(event, UsageReport)]
