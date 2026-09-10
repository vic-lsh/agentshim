"""Claude Code ``stream-json`` parsing."""

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
from agentshim.providers.claude import ClaudeStreamParser


def _parser(*, expect_structured: bool = False) -> tuple[ClaudeStreamParser, list[AgentEvent]]:
    events: list[AgentEvent] = []
    return ClaudeStreamParser(events.append, expect_structured=expect_structured), events


def _line(payload: dict[str, Any]) -> str:
    return json.dumps(payload) + "\n"


def _assistant(*blocks: dict[str, Any], usage: dict[str, Any] | None = None) -> str:
    message: dict[str, Any] = {"role": "assistant", "content": list(blocks)}
    if usage is not None:
        message["usage"] = usage
    return _line({"type": "assistant", "message": message})


class TestSessionId:
    def test_system_init_starts_the_session(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(_line({"type": "system", "subtype": "init", "session_id": "abc-123"}))
        assert events == [SessionStarted("abc-123")]
        assert parser.finish().session_id == "abc-123"

    def test_a_system_frame_without_an_id_emits_nothing(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(_line({"type": "system", "subtype": "init"}))
        assert events == []
        assert parser.finish().session_id is None

    def test_only_the_first_id_is_kept(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(_line({"type": "system", "subtype": "init", "session_id": "first"}))
        parser.feed_stdout(_line({"type": "system", "subtype": "init", "session_id": "second"}))
        assert events == [SessionStarted("first")]
        assert parser.finish().session_id == "first"


class TestAssistantContent:
    def test_text_blocks_emit_and_accumulate(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(_assistant({"type": "text", "text": "Hello\n"}))
        parser.feed_stdout(_assistant({"type": "text", "text": "World"}))
        assert events == [AssistantText("Hello\n"), AssistantText("World")]
        assert parser.finish().text == "Hello\n\nWorld"

    def test_thinking_blocks_become_reasoning(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(_assistant({"type": "thinking", "thinking": "hmm"}))
        assert events == [Reasoning("hmm")]
        assert parser.finish().text == ""

    def test_tool_use_blocks_emit_a_tool_call(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(_assistant({"type": "tool_use", "id": "t1", "name": "Bash", "input": {"cmd": "ls"}}))
        assert events == [ToolCall("t1", "Bash", {"cmd": "ls"})]

    def test_multiple_blocks_are_emitted_in_order(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(
            _assistant(
                {"type": "text", "text": "running"},
                {"type": "tool_use", "id": "t1", "name": "Read", "input": {"path": "/x"}},
            )
        )
        assert [type(event).__name__ for event in events] == ["AssistantText", "ToolCall"]

    def test_an_unnamed_tool_falls_back(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(_assistant({"type": "tool_use", "id": "t1", "input": {}}))
        assert isinstance(events[0], ToolCall)
        assert events[0].tool == "Tool"

    def test_unknown_block_types_are_ignored(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(_assistant({"type": "image", "source": {}}))
        assert events == []

    def test_a_message_without_content_or_usage_is_ignored(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(_line({"type": "assistant", "message": {"role": "assistant"}}))
        assert events == []


class TestToolResults:
    def test_a_result_is_paired_with_its_call(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(_assistant({"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}))
        parser.feed_stdout(
            _line(
                {
                    "type": "user",
                    "message": {"content": [{"type": "tool_result", "tool_use_id": "t1", "content": "file.txt"}]},
                }
            )
        )
        result = events[-1]
        assert isinstance(result, ToolResult)
        assert result.tool == "Bash"
        assert result.stdout == "file.txt"
        assert result.duration_s is not None
        assert result.duration_s >= 0

    def test_an_unpaired_result_has_no_duration(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(
            _line({"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "t9"}]}})
        )
        result = events[-1]
        assert isinstance(result, ToolResult)
        assert result.tool == "Tool"
        assert result.duration_s is None

    def test_list_content_is_flattened(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(
            _line(
                {
                    "type": "user",
                    "message": {"content": [{"type": "tool_result", "tool_use_id": "t1", "content": ["a", "b"]}]},
                }
            )
        )
        result = events[-1]
        assert isinstance(result, ToolResult)
        assert result.stdout == "a\nb"

    def test_an_error_result_lands_on_stderr(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(
            _line(
                {
                    "type": "user",
                    "message": {
                        "content": [
                            {"type": "tool_result", "tool_use_id": "t1", "content": "boom", "is_error": True},
                        ]
                    },
                }
            )
        )
        result = events[-1]
        assert isinstance(result, ToolResult)
        assert result.stdout == ""
        assert result.stderr == "boom"
        assert result.exit_code == 1

    def test_a_user_message_without_a_tool_result_is_ignored(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(_line({"type": "user", "message": {"content": [{"type": "text", "text": "hi"}]}}))
        assert events == []


class TestUsage:
    def test_cache_tokens_fold_into_input_tokens(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(
            _line(
                {
                    "type": "result",
                    "result": "done",
                    "num_turns": 4,
                    "total_cost_usd": 0.5,
                    "usage": {
                        "input_tokens": 100,
                        "output_tokens": 40,
                        "cache_creation_input_tokens": 30,
                        "cache_read_input_tokens": 20,
                    },
                }
            )
        )
        parsed = parser.finish()
        tokens = parsed.usage.tokens
        assert tokens.input_tokens == 150
        assert tokens.output_tokens == 40
        assert tokens.cached_input_tokens == 50
        assert tokens.cache_write_input_tokens == 30
        assert tokens.turns == 4
        assert tokens.cached_input_tokens <= tokens.input_tokens
        assert parsed.usage.provider == "claude"
        assert parsed.usage.total_cost_usd == 0.5
        assert parsed.cost_usd == 0.5
        assert isinstance(events[-1], UsageReport)

    def test_the_raw_mapping_is_preserved(self) -> None:
        parser, _ = _parser()
        raw = {"input_tokens": 7000, "output_tokens": 120}
        parser.feed_stdout(_line({"type": "result", "result": "done", "usage": raw}))
        assert parser.finish().usage.raw == raw

    def test_a_result_without_usage_degrades_to_zero(self) -> None:
        parser, _ = _parser()
        parser.feed_stdout(_line({"type": "result", "result": "done"}))
        parsed = parser.finish()
        assert parsed.usage.tokens.input_tokens == 0
        assert parsed.usage.provider == "claude"
        assert parsed.usage.raw is None
        assert parsed.cost_usd is None

    def test_assistant_usage_is_reported_incrementally(self) -> None:
        parser, events = _parser()
        usage = {
            "input_tokens": 14000,
            "output_tokens": 50,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }
        parser.feed_stdout(_assistant({"type": "text", "text": "ok"}, usage=usage))
        report = events[0]
        assert isinstance(report, UsageReport)
        assert report.usage.tokens.input_tokens == 14000
        assert report.cost_usd is None
        assert report.usage.raw == usage

    def test_no_usage_report_when_the_assistant_frame_has_none(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(_assistant({"type": "text", "text": "ok"}))
        assert not any(isinstance(event, UsageReport) for event in events)

    def test_no_usage_at_all_still_yields_a_claude_provider_usage(self) -> None:
        parser, _ = _parser()
        assert parser.finish().usage.provider == "claude"


class TestFinalText:
    def test_the_result_frame_wins_over_accumulated_text(self) -> None:
        parser, _ = _parser()
        parser.feed_stdout(_assistant({"type": "text", "text": "streamed"}))
        parser.feed_stdout(_line({"type": "result", "result": "final answer"}))
        assert parser.finish().text == "final answer"

    def test_accumulated_text_is_used_without_a_result_frame(self) -> None:
        parser, _ = _parser()
        parser.feed_stdout(_assistant({"type": "text", "text": "streamed"}))
        assert parser.finish().text == "streamed"

    def test_an_empty_stream_yields_empty_text(self) -> None:
        parser, _ = _parser()
        assert parser.finish().text == ""


class TestStructuredOutput:
    def test_structured_output_is_captured_from_the_result_frame(self) -> None:
        parser, _ = _parser(expect_structured=True)
        parser.feed_stdout(
            _line({"type": "result", "result": '{"a":3}', "structured_output": {"a": 3, "b": 5}}),
        )
        parsed = parser.finish()
        assert parsed.structured_output == {"a": 3, "b": 5}
        assert parsed.text == '{"a":3}'

    def test_a_null_structured_output_is_ignored(self) -> None:
        parser, _ = _parser()
        parser.feed_stdout(_line({"type": "result", "result": "done", "structured_output": None}))
        assert parser.finish().structured_output is None

    def test_absent_structured_output_stays_none_when_not_expected(self) -> None:
        parser, _ = _parser()
        parser.feed_stdout(_line({"type": "result", "result": '{"a": 3}'}))
        assert parser.finish().structured_output is None

    def test_a_json_result_is_used_when_a_schema_was_requested(self) -> None:
        parser, _ = _parser(expect_structured=True)
        parser.feed_stdout(_line({"type": "result", "result": '{"a": 3}'}))
        assert parser.finish().structured_output == {"a": 3}

    def test_prose_is_not_forced_into_structured_output(self) -> None:
        parser, _ = _parser(expect_structured=True)
        parser.feed_stdout(_line({"type": "result", "result": "here is your answer"}))
        assert parser.finish().structured_output is None


class TestMalformedLines:
    def test_a_non_json_line_becomes_raw_output(self) -> None:
        parser, events = _parser()
        parser.feed_stdout("starting claude...\n")
        assert events == [RawOutput("starting claude...")]

    def test_a_non_object_json_line_does_not_crash(self) -> None:
        """A bare scalar or array on stdout used to blow up ``data.get``."""
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
        parser.feed_stdout(_line({"type": "control_response", "id": 1}))
        assert events == []

    def test_a_frame_without_a_type_is_ignored(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(_line({"session_id": "x"}))
        assert events == []


class TestStderr:
    def test_stderr_lines_are_emitted(self) -> None:
        parser, events = _parser()
        parser.feed_stderr("warning: something\n")
        assert events == [Stderr("warning: something")]

    def test_blank_stderr_is_dropped(self) -> None:
        parser, events = _parser()
        parser.feed_stderr("\n")
        assert events == []


class TestErrorResults:
    def test_an_error_result_emits_a_provider_error(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(
            _line({"type": "result", "subtype": "error_max_turns", "is_error": True, "result": "turn limit"})
        )
        assert ProviderError("turn limit") in events
        assert parser.finish().error == "turn limit"

    def test_the_subtype_is_used_when_there_is_no_message(self) -> None:
        parser, _ = _parser()
        parser.feed_stdout(_line({"type": "result", "subtype": "error_during_execution", "is_error": True}))
        assert parser.finish().error == "error_during_execution"
