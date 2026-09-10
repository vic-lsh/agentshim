"""Gemini CLI ``stream-json`` parsing."""

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
from agentshim.providers.gemini import GeminiStreamParser


def _parser() -> tuple[GeminiStreamParser, list[AgentEvent]]:
    events: list[AgentEvent] = []
    return GeminiStreamParser(events.append), events


def _line(payload: dict[str, Any]) -> str:
    return json.dumps({**payload, "timestamp": "2026-01-01T00:00:00.000Z"}) + "\n"


def _stats(**overrides: int) -> dict[str, int]:
    base = {
        "total_tokens": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cached": 0,
        "input": 0,
        "duration_ms": 0,
        "tool_calls": 0,
    }
    base.update(overrides)
    return base


class TestSessionId:
    def test_the_init_frame_starts_the_session(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(
            _line({"type": "init", "session_id": "abc-123", "model": "gemini-2.5-pro"})
        )
        assert events == [SessionStarted("abc-123")]
        assert parser.finish().session_id == "abc-123"

    def test_an_init_frame_without_an_id_emits_nothing(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(_line({"type": "init", "model": "gemini-2.5-pro"}))
        assert events == []
        assert parser.finish().session_id is None

    def test_only_the_first_id_is_kept(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(_line({"type": "init", "session_id": "first"}))
        parser.feed_stdout(_line({"type": "init", "session_id": "second"}))
        assert events == [SessionStarted("first")]
        assert parser.finish().session_id == "first"


class TestAssistantText:
    def test_assistant_deltas_emit_and_concatenate(self) -> None:
        """Each assistant frame is a chunk of one message, not a message."""
        parser, events = _parser()
        parser.feed_stdout(
            _line({"type": "message", "role": "assistant", "content": "Hello, ", "delta": True})
        )
        parser.feed_stdout(
            _line({"type": "message", "role": "assistant", "content": "world", "delta": True})
        )
        assert events == [AssistantText("Hello, "), AssistantText("world")]
        assert parser.finish().text == "Hello, world"

    def test_the_echoed_user_prompt_is_not_model_output(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(_line({"type": "message", "role": "user", "content": "what is 2 + 2?"}))
        assert events == []
        assert parser.finish().text == ""

    def test_empty_content_is_dropped(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(
            _line({"type": "message", "role": "assistant", "content": "", "delta": True})
        )
        assert events == []

    def test_no_reasoning_is_reported(self) -> None:
        """Gemini CLI 0.26.0 drops thought events before the stream formatter."""
        parser, events = _parser()
        parser.feed_stdout(_line({"type": "message", "role": "assistant", "content": "answer"}))
        parser.feed_stdout(_line({"type": "result", "status": "success", "stats": _stats()}))
        assert not any(isinstance(event, Reasoning) for event in events)


class TestTools:
    def test_a_tool_call_is_reported(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(
            _line(
                {
                    "type": "tool_use",
                    "tool_name": "run_shell_command",
                    "tool_id": "c1",
                    "parameters": {"cmd": "ls"},
                }
            )
        )
        assert events == [ToolCall("c1", "run_shell_command", {"cmd": "ls"})]

    def test_an_unnamed_tool_falls_back(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(_line({"type": "tool_use", "tool_id": "c1", "parameters": {}}))
        assert isinstance(events[0], ToolCall)
        assert events[0].tool == "Tool"

    def test_a_result_is_paired_with_its_call(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(
            _line({"type": "tool_use", "tool_name": "read_file", "tool_id": "c1", "parameters": {}})
        )
        parser.feed_stdout(
            _line(
                {"type": "tool_result", "tool_id": "c1", "status": "success", "output": "file.txt"}
            )
        )
        result = events[-1]
        assert isinstance(result, ToolResult)
        assert result.tool == "read_file"
        assert result.stdout == "file.txt"
        assert result.exit_code is None
        assert result.duration_s is not None
        assert result.duration_s >= 0

    def test_an_unpaired_result_has_no_duration(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(
            _line({"type": "tool_result", "tool_id": "c9", "status": "success", "output": "x"})
        )
        result = events[-1]
        assert isinstance(result, ToolResult)
        assert result.tool == "Tool"
        assert result.duration_s is None

    def test_a_failed_tool_lands_on_stderr(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(
            _line(
                {
                    "type": "tool_result",
                    "tool_id": "c1",
                    "status": "error",
                    "error": {"type": "TOOL_EXECUTION_ERROR", "message": "boom"},
                }
            )
        )
        result = events[-1]
        assert isinstance(result, ToolResult)
        assert result.stdout == ""
        assert result.stderr == "boom"
        assert result.exit_code == 1


class TestUsage:
    def test_stats_become_a_usage_report(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(
            _line(
                {
                    "type": "result",
                    "status": "success",
                    "stats": _stats(
                        total_tokens=190, input_tokens=150, output_tokens=40, cached=50, input=100
                    ),
                }
            )
        )
        parsed = parser.finish()
        tokens = parsed.usage.tokens
        assert tokens.input_tokens == 150
        assert tokens.output_tokens == 40
        assert tokens.cached_input_tokens == 50
        assert tokens.turns == 1
        assert parsed.usage.provider == "gemini"
        assert isinstance(events[-1], UsageReport)

    def test_cached_never_exceeds_input(self) -> None:
        """The clamp holds even if a build reports the two disjointly."""
        parser, _ = _parser()
        parser.feed_stdout(
            _line(
                {"type": "result", "status": "success", "stats": _stats(input_tokens=10, cached=99)}
            )
        )
        tokens = parser.finish().usage.tokens
        assert tokens.cached_input_tokens <= tokens.input_tokens

    def test_the_raw_stats_are_preserved(self) -> None:
        parser, _ = _parser()
        stats = _stats(input_tokens=7000, output_tokens=120)
        parser.feed_stdout(_line({"type": "result", "status": "success", "stats": stats}))
        assert parser.finish().usage.raw == stats

    def test_a_result_without_stats_degrades_to_zero(self) -> None:
        parser, _ = _parser()
        parser.feed_stdout(_line({"type": "result", "status": "success"}))
        parsed = parser.finish()
        assert parsed.usage.tokens.input_tokens == 0
        assert parsed.usage.provider == "gemini"
        assert parsed.usage.raw is None

    def test_gemini_reports_no_cost(self) -> None:
        parser, _ = _parser()
        parser.feed_stdout(
            _line({"type": "result", "status": "success", "stats": _stats(input_tokens=5)})
        )
        assert parser.finish().cost_usd is None

    def test_no_stream_at_all_still_yields_a_gemini_provider_usage(self) -> None:
        parser, _ = _parser()
        assert parser.finish().usage.provider == "gemini"


class TestErrors:
    def test_an_error_frame_is_reported(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(
            _line(
                {"type": "error", "severity": "error", "message": "Maximum session turns exceeded"}
            )
        )
        assert ProviderError("Maximum session turns exceeded") in events
        assert parser.finish().error == "Maximum session turns exceeded"

    def test_a_warning_is_surfaced_without_failing_the_turn(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(
            _line({"type": "error", "severity": "warning", "message": "Loop detected"})
        )
        assert ProviderError("Loop detected") in events
        assert parser.finish().error is None

    def test_a_failed_result_carries_the_fatal_message(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(
            _line(
                {
                    "type": "result",
                    "status": "error",
                    "error": {
                        "type": "FatalTurnLimitedError",
                        "message": "Reached max session turns",
                    },
                    "stats": _stats(),
                }
            )
        )
        assert ProviderError("Reached max session turns") in events
        assert parser.finish().error == "Reached max session turns"

    def test_a_failed_result_without_a_message_still_errors(self) -> None:
        parser, _ = _parser()
        parser.feed_stdout(_line({"type": "result", "status": "error"}))
        assert parser.finish().error == "gemini reported an error"


class TestMalformedLines:
    def test_a_non_json_line_becomes_raw_output(self) -> None:
        parser, events = _parser()
        parser.feed_stdout("Loaded cached credentials.\n")
        assert events == [RawOutput("Loaded cached credentials.")]

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
        parser.feed_stdout(_line({"type": "user_feedback", "message": "hi"}))
        assert events == []

    def test_a_frame_without_a_type_is_ignored(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(_line({"session_id": "x"}))
        assert events == []


class TestStderr:
    def test_stderr_lines_are_emitted(self) -> None:
        parser, events = _parser()
        parser.feed_stderr("[WARNING] YOLO mode is enabled.\n")
        assert events == [Stderr("[WARNING] YOLO mode is enabled.")]

    def test_blank_stderr_is_dropped(self) -> None:
        parser, events = _parser()
        parser.feed_stderr("\n")
        assert events == []
