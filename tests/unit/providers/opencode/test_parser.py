"""opencode ``run --format json`` parsing."""

from __future__ import annotations

import json
from typing import Any

from agentshim import (
    AssistantText,
    ProviderError,
    RawOutput,
    Reasoning,
    SessionStarted,
    Stderr,
    ToolCall,
    ToolResult,
    UsageReport,
)
from agentshim.core.events import AgentEvent
from agentshim.providers.opencode import OpencodeStreamParser

SESSION = "ses_f7612ae8bffeivvPisW3kRh64e"


def _parser() -> tuple[OpencodeStreamParser, list[AgentEvent]]:
    events: list[AgentEvent] = []
    return OpencodeStreamParser(events.append), events


def _line(
    kind: str,
    part: dict[str, Any],
    *,
    session_id: str | None = SESSION,
    error: dict[str, Any] | None = None,
) -> str:
    payload: dict[str, Any] = {"type": kind, "timestamp": 1789020428786}
    if error is not None:
        payload["error"] = error
    if session_id is not None:
        payload["sessionID"] = session_id
    if part:
        payload["part"] = {"id": "prt_1", "messageID": "msg_1", "sessionID": session_id, **part}
    return json.dumps(payload) + "\n"


def _tokens(
    *,
    input_tokens: int = 0,
    output: int = 0,
    reasoning: int = 0,
    cache: dict[str, int] | None = None,
) -> dict[str, Any]:
    read_write = cache if cache is not None else {"read": 0, "write": 0}
    return {
        "total": input_tokens + output,
        "input": input_tokens,
        "output": output,
        "reasoning": reasoning,
        "cache": read_write,
    }


class TestSessionId:
    def test_the_first_frame_starts_the_session(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(_line("step_start", {"type": "step-start"}))
        assert events == [SessionStarted(SESSION)]
        assert parser.finish().session_id == SESSION

    def test_a_frame_without_a_session_id_emits_nothing(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(_line("step_start", {"type": "step-start"}, session_id=None))
        assert events == []
        assert parser.finish().session_id is None

    def test_only_the_first_id_is_kept(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(_line("step_start", {"type": "step-start"}))
        parser.feed_stdout(_line("step_start", {"type": "step-start"}, session_id="ses_other"))
        assert events == [SessionStarted(SESSION)]
        assert parser.finish().session_id == SESSION


class TestAssistantText:
    def test_text_parts_emit_and_accumulate(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(_line("text", {"type": "text", "text": "Hello"}))
        parser.feed_stdout(_line("text", {"type": "text", "text": "World"}))
        assert [event for event in events if isinstance(event, AssistantText)] == [
            AssistantText("Hello"),
            AssistantText("World"),
        ]
        assert parser.finish().text == "Hello\nWorld"

    def test_empty_text_is_dropped(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(_line("text", {"type": "text", "text": ""}, session_id=None))
        assert events == []

    def test_reasoning_parts_become_reasoning(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(
            _line("reasoning", {"type": "reasoning", "text": "hmm"}, session_id=None)
        )
        assert events == [Reasoning("hmm")]
        assert parser.finish().text == ""

    def test_step_start_reports_nothing_of_its_own(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(_line("step_start", {"type": "step-start"}, session_id=None))
        assert events == []


class TestTools:
    def test_a_completed_tool_emits_a_call_and_a_result(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(
            _line(
                "tool_use",
                {
                    "type": "tool",
                    "callID": "call_1",
                    "tool": "bash",
                    "state": {
                        "status": "completed",
                        "input": {"command": "ls"},
                        "output": "file.txt",
                        "time": {"start": 1000, "end": 2500},
                    },
                },
                session_id=None,
            )
        )
        call, result = events
        assert call == ToolCall("call_1", "bash", {"command": "ls"})
        assert isinstance(result, ToolResult)
        assert result.tool == "bash"
        assert result.stdout == "file.txt"
        assert result.exit_code is None
        assert result.duration_s == 1.5

    def test_the_part_time_block_supplies_the_duration(self) -> None:
        """0.5 dropped completed tools entirely by checking for status "success"."""
        parser, events = _parser()
        parser.feed_stdout(
            _line(
                "tool_use",
                {
                    "type": "tool",
                    "callID": "c1",
                    "tool": "read",
                    "state": {"status": "completed", "input": {}, "output": "x"},
                },
                session_id=None,
            )
        )
        result = events[-1]
        assert isinstance(result, ToolResult)
        assert result.duration_s is not None
        assert result.duration_s >= 0

    def test_a_failed_tool_lands_on_stderr(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(
            _line(
                "tool_use",
                {
                    "type": "tool",
                    "callID": "c1",
                    "tool": "bash",
                    "state": {
                        "status": "error",
                        "input": {},
                        "error": "boom",
                        "time": {"start": 0, "end": 500},
                    },
                },
                session_id=None,
            )
        )
        result = events[-1]
        assert isinstance(result, ToolResult)
        assert result.stdout == ""
        assert result.stderr == "boom"
        assert result.exit_code == 1

    def test_a_pending_tool_is_not_reported_yet(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(
            _line(
                "tool_use",
                {
                    "type": "tool",
                    "callID": "c1",
                    "tool": "bash",
                    "state": {"status": "pending", "input": {}},
                },
                session_id=None,
            )
        )
        assert events == []

    def test_an_unnamed_tool_falls_back(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(
            _line(
                "tool_use",
                {
                    "type": "tool",
                    "callID": "c1",
                    "state": {"status": "completed", "input": {}, "output": ""},
                },
                session_id=None,
            )
        )
        assert isinstance(events[0], ToolCall)
        assert events[0].tool == "Tool"


class TestUsage:
    def test_cache_tokens_fold_into_input_tokens(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(
            _line(
                "step_finish",
                {
                    "type": "step-finish",
                    "reason": "stop",
                    "cost": 0.25,
                    "tokens": _tokens(
                        input_tokens=100, output=30, reasoning=10, cache={"read": 40, "write": 10}
                    ),
                },
                session_id=None,
            )
        )
        parsed = parser.finish()
        tokens = parsed.usage.tokens
        assert tokens.input_tokens == 150
        assert tokens.output_tokens == 40
        assert tokens.cached_input_tokens == 50
        assert tokens.cache_write_input_tokens == 10
        assert tokens.reasoning_output_tokens == 10
        assert tokens.turns == 1
        assert tokens.cached_input_tokens <= tokens.input_tokens
        assert parsed.usage.provider == "opencode"
        assert parsed.cost_usd == 0.25
        assert isinstance(events[-1], UsageReport)

    def test_steps_accumulate(self) -> None:
        parser, _ = _parser()
        step = _line(
            "step_finish",
            {
                "type": "step-finish",
                "reason": "stop",
                "cost": 0.5,
                "tokens": _tokens(input_tokens=10, output=5),
            },
            session_id=None,
        )
        parser.feed_stdout(step)
        parser.feed_stdout(step)
        parsed = parser.finish()
        assert parsed.usage.tokens.input_tokens == 20
        assert parsed.usage.tokens.turns == 2
        assert parsed.cost_usd == 1.0

    def test_the_raw_tokens_are_preserved(self) -> None:
        parser, _ = _parser()
        tokens = _tokens(input_tokens=7000, output=120)
        parser.feed_stdout(
            _line(
                "step_finish", {"type": "step-finish", "cost": 0, "tokens": tokens}, session_id=None
            )
        )
        assert parser.finish().usage.raw == tokens

    def test_a_step_without_tokens_degrades_to_zero(self) -> None:
        parser, _ = _parser()
        parser.feed_stdout(
            _line("step_finish", {"type": "step-finish", "reason": "stop"}, session_id=None)
        )
        parsed = parser.finish()
        assert parsed.usage.tokens.input_tokens == 0
        assert parsed.usage.tokens.turns == 1
        assert parsed.cost_usd is None

    def test_no_stream_at_all_still_yields_an_opencode_provider_usage(self) -> None:
        parser, _ = _parser()
        assert parser.finish().usage.provider == "opencode"


class TestErrors:
    def test_a_session_error_is_reported(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(
            _line(
                "error",
                {},
                session_id=None,
                error={"name": "ProviderAuthError", "data": {"message": "no key"}},
            )
        )
        assert ProviderError("no key") in events
        assert parser.finish().error == "no key"

    def test_the_error_name_is_used_without_a_message(self) -> None:
        parser, _ = _parser()
        parser.feed_stdout(_line("error", {}, session_id=None, error={"name": "UnknownError"}))
        assert parser.finish().error == "UnknownError"

    def test_an_unrecognizable_error_still_reports_something(self) -> None:
        parser, _ = _parser()
        parser.feed_stdout(_line("error", {}, session_id=None, error={}))
        assert parser.finish().error == "opencode reported an error"


class TestMalformedLines:
    def test_a_non_json_line_becomes_raw_output(self) -> None:
        parser, events = _parser()
        parser.feed_stdout("starting opencode...\n")
        assert events == [RawOutput("starting opencode...")]

    def test_a_non_object_json_line_does_not_crash(self) -> None:
        parser, events = _parser()
        for line in ("42\n", "[1, 2]\n", '"hello"\n', "null\n", "true\n"):
            parser.feed_stdout(line)
        assert [type(event).__name__ for event in events] == ["RawOutput"] * 5
        assert parser.finish().text == ""

    def test_blank_lines_are_dropped(self) -> None:
        parser, events = _parser()
        parser.feed_stdout("\n")
        parser.feed_stdout("")
        assert events == []

    def test_an_unknown_frame_type_is_ignored(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(_line("snapshot", {"type": "snapshot"}, session_id=None))
        assert events == []

    def test_a_frame_without_a_type_is_ignored(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(json.dumps({"timestamp": 1}) + "\n")
        assert events == []


class TestStderr:
    def test_stderr_lines_are_emitted(self) -> None:
        parser, events = _parser()
        parser.feed_stderr("WARN missing config\n")
        assert events == [Stderr("WARN missing config")]

    def test_blank_stderr_is_dropped(self) -> None:
        parser, events = _parser()
        parser.feed_stderr("\n")
        assert events == []
