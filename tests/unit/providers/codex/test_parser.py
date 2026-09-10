"""Codex ``--json`` stream parsing."""

from __future__ import annotations

import json
from typing import Any

from agentshim import (
    AssistantText,
    Lifecycle,
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
from agentshim.providers.codex import CodexStreamParser


def _parser(*, expect_structured: bool = False) -> tuple[CodexStreamParser, list[AgentEvent]]:
    events: list[AgentEvent] = []
    return CodexStreamParser(events.append, expect_structured=expect_structured), events


def _line(payload: dict[str, Any]) -> str:
    return json.dumps(payload) + "\n"


def _item(event_type: str, item: dict[str, Any]) -> str:
    return _line({"type": event_type, "item": item})


def _message(text: str) -> str:
    return _item("item.completed", {"id": "m1", "type": "agent_message", "text": text})


def _usage_line(input_tokens: int, cached: int, output: int) -> str:
    return _line(
        {
            "type": "turn.completed",
            "usage": {
                "input_tokens": input_tokens,
                "cached_input_tokens": cached,
                "output_tokens": output,
            },
        }
    )


class TestThread:
    def test_thread_started_names_the_session(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(_line({"type": "thread.started", "thread_id": "t-1"}))
        assert events == [SessionStarted("t-1"), Lifecycle("thread_started", "t-1")]
        assert parser.finish().session_id == "t-1"

    def test_a_thread_frame_without_an_id_emits_nothing(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(_line({"type": "thread.started"}))
        assert events == []
        assert parser.finish().session_id is None

    def test_only_the_first_id_is_kept(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(_line({"type": "thread.started", "thread_id": "first"}))
        parser.feed_stdout(_line({"type": "thread.started", "thread_id": "second"}))
        assert parser.finish().session_id == "first"
        assert [event for event in events if isinstance(event, SessionStarted)] == [
            SessionStarted("first")
        ]

    def test_turn_started_is_lifecycle_plumbing(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(_line({"type": "turn.started"}))
        assert events == [Lifecycle("turn_started", "")]


class TestMessages:
    def test_an_agent_message_is_assistant_text(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(_message("final answer"))
        assert events == [AssistantText("final answer")]
        assert parser.finish().text == "final answer"

    def test_the_last_agent_message_is_the_answer(self) -> None:
        """Codex has no terminal result frame; the last message is the reply."""
        parser, _ = _parser()
        for text in ("first", "second", "third"):
            parser.feed_stdout(_message(text))
        assert parser.finish().text == "third"

    def test_a_started_message_item_emits_nothing(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(_item("item.started", {"id": "m1", "type": "agent_message", "text": ""}))
        assert events == []

    def test_reasoning_becomes_a_reasoning_event(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(
            _item("item.completed", {"id": "r1", "type": "reasoning", "text": "grep for X"})
        )
        assert events == [Reasoning("grep for X")]
        assert parser.finish().text == ""

    def test_an_empty_stream_yields_empty_text(self) -> None:
        parser, _ = _parser()
        assert parser.finish().text == ""


class TestCommandExecution:
    def test_a_command_pairs_into_a_call_and_a_result(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(
            _item("item.started", {"id": "c1", "type": "command_execution", "command": "ls -la"}),
        )
        parser.feed_stdout(
            _item(
                "item.completed",
                {
                    "id": "c1",
                    "type": "command_execution",
                    "command": "ls -la",
                    "aggregated_output": "file.txt\n",
                    "exit_code": 0,
                },
            )
        )
        call, result = events
        assert call == ToolCall("c1", "execute", {"command": "ls -la"})
        assert isinstance(result, ToolResult)
        assert (result.tool, result.stdout, result.exit_code) == ("execute", "file.txt\n", 0)
        assert result.duration_s is not None
        assert result.duration_s >= 0

    def test_a_failing_command_reports_on_stderr_with_its_exit_code(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(
            _item(
                "item.completed",
                {
                    "id": "c2",
                    "type": "command_execution",
                    "command": "false",
                    "aggregated_output": "no such file\n",
                    "exit_code": 1,
                },
            )
        )
        result = events[-1]
        assert isinstance(result, ToolResult)
        assert result.exit_code == 1
        assert result.stderr == "no such file\n"
        assert result.stdout == ""

    def test_an_unpaired_completion_has_no_duration(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(
            _item("item.completed", {"id": "c9", "type": "command_execution", "command": "ls"})
        )
        result = events[-1]
        assert isinstance(result, ToolResult)
        assert result.duration_s is None


class TestGenericItems:
    def test_an_mcp_tool_call_reports_its_arguments_then_its_result(self) -> None:
        parser, events = _parser()
        started = {
            "id": "item_5",
            "type": "mcp_tool_call",
            "status": "in_progress",
            "server": "docs",
            "tool": "search",
            "arguments": {"query": "x"},
            "result": None,
        }
        parser.feed_stdout(_item("item.started", started))
        parser.feed_stdout(
            _item(
                "item.completed",
                {
                    **started,
                    "status": "completed",
                    "result": {"content": [{"type": "text", "text": "hits"}]},
                },
            )
        )
        call, result = events
        assert isinstance(call, ToolCall)
        assert call.tool == "mcp_tool_call"
        assert call.args == {
            "server": "docs",
            "tool": "search",
            "arguments": {"query": "x"},
            "result": None,
        }
        assert isinstance(result, ToolResult)
        assert result.tool == "mcp_tool_call"
        assert "hits" in result.stdout

    def test_a_failed_item_surfaces_its_error_payload_on_stderr(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(
            _item(
                "item.completed",
                {
                    "id": "m2",
                    "type": "mcp_tool_call",
                    "status": "failed",
                    "error": {"message": "boom"},
                },
            )
        )
        result = events[-1]
        assert isinstance(result, ToolResult)
        assert "boom" in result.stderr
        assert result.stdout == ""
        assert result.exit_code == 1

    def test_a_file_change_is_a_tool_call_with_no_output(self) -> None:
        parser, events = _parser()
        item = {
            "id": "f1",
            "type": "file_change",
            "status": "completed",
            "path": "engine.py",
            "kind": "update",
        }
        parser.feed_stdout(_item("item.started", item))
        parser.feed_stdout(_item("item.completed", item))
        call, result = events
        assert isinstance(call, ToolCall)
        assert call.args == {"path": "engine.py", "kind": "update"}
        assert isinstance(result, ToolResult)
        assert result.stdout == ""

    def test_items_pair_by_id_not_by_argument_equality(self) -> None:
        parser, events = _parser()
        for kind, item_id in (
            ("item.started", "a"),
            ("item.started", "b"),
            ("item.completed", "b"),
        ):
            parser.feed_stdout(_item(kind, {"id": item_id, "type": "web_search", "query": "docs"}))
        assert [type(event).__name__ for event in events] == ["ToolCall", "ToolCall", "ToolResult"]
        result = events[-1]
        assert isinstance(result, ToolResult)
        assert result.tool_id == "b"

    def test_an_untyped_item_still_reports_as_a_tool(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(_item("item.started", {"id": "x1", "detail": "?"}))
        call = events[0]
        assert isinstance(call, ToolCall)
        assert call.tool == "item"

    def test_an_item_frame_without_an_item_is_ignored(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(_line({"type": "item.started"}))
        assert events == []


class TestUsage:
    def test_codex_input_tokens_already_include_the_cached_prefix(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(_usage_line(1200, 800, 150))
        parsed = parser.finish()
        tokens = parsed.usage.tokens
        assert tokens.input_tokens == 1200
        assert tokens.cached_input_tokens == 800
        assert tokens.output_tokens == 150
        assert tokens.turns == 1
        assert tokens.cached_input_tokens <= tokens.input_tokens
        assert parsed.usage.provider == "codex"
        assert parsed.cost_usd is None
        assert isinstance(events[-1], UsageReport)
        assert Lifecycle("turn_completed", "in=1200 cached=800 out=150") in events

    def test_repeated_turns_accumulate(self) -> None:
        parser, _ = _parser()
        parser.feed_stdout(_usage_line(100, 10, 5))
        parser.feed_stdout(_usage_line(200, 20, 7))
        tokens = parser.finish().usage.tokens
        assert (tokens.input_tokens, tokens.cached_input_tokens, tokens.output_tokens) == (
            300,
            30,
            12,
        )
        assert tokens.turns == 2

    def test_the_raw_mapping_is_preserved(self) -> None:
        parser, _ = _parser()
        parser.feed_stdout(_usage_line(7, 0, 1))
        assert parser.finish().usage.raw == {
            "input_tokens": 7,
            "cached_input_tokens": 0,
            "output_tokens": 1,
        }

    def test_a_turn_without_usage_reports_no_tokens(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(_line({"type": "turn.completed"}))
        parsed = parser.finish()
        assert parsed.usage.tokens.input_tokens == 0
        assert parsed.usage.raw is None
        assert events == [Lifecycle("turn_completed", "")]

    def test_no_usage_at_all_still_yields_a_codex_provider_usage(self) -> None:
        parser, _ = _parser()
        assert parser.finish().usage.provider == "codex"


class TestStructuredOutput:
    def test_the_final_message_is_decoded_when_a_schema_was_requested(self) -> None:
        parser, _ = _parser(expect_structured=True)
        parser.feed_stdout(_message('{"answer": 4}'))
        parsed = parser.finish()
        assert parsed.structured_output == {"answer": 4}
        assert parsed.text == '{"answer": 4}'

    def test_prose_is_not_forced_into_structured_output(self) -> None:
        parser, _ = _parser(expect_structured=True)
        parser.feed_stdout(_message("here is your answer"))
        assert parser.finish().structured_output is None

    def test_json_text_stays_text_when_no_schema_was_requested(self) -> None:
        parser, _ = _parser()
        parser.feed_stdout(_message('{"answer": 4}'))
        assert parser.finish().structured_output is None

    def test_an_empty_stream_has_no_structured_output(self) -> None:
        parser, _ = _parser(expect_structured=True)
        assert parser.finish().structured_output is None


class TestErrors:
    def test_turn_failed_is_a_provider_error(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(_line({"type": "turn.failed", "error": {"message": "rate limited"}}))
        assert events == [ProviderError("rate limited")]
        assert parser.finish().error == "rate limited"

    def test_a_top_level_error_is_a_provider_error(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(_line({"type": "error", "message": "stream disconnected"}))
        assert events == [ProviderError("stream disconnected")]
        assert parser.finish().error == "stream disconnected"

    def test_a_failure_without_a_message_still_reports(self) -> None:
        parser, _ = _parser()
        parser.feed_stdout(_line({"type": "turn.failed"}))
        assert parser.finish().error == "codex reported an error"


class TestMalformedLines:
    def test_a_non_json_line_becomes_raw_output(self) -> None:
        parser, events = _parser()
        parser.feed_stdout("starting codex 1.2.3\n")
        assert events == [RawOutput("starting codex 1.2.3")]

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
        parser.feed_stdout(
            _line({"type": "item.updated", "item": {"type": "reasoning", "delta": "..."}})
        )
        assert events == []

    def test_a_frame_without_a_type_is_ignored(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(_line({"thread_id": "x"}))
        assert events == []


class TestStderr:
    def test_stderr_lines_are_emitted_as_stderr_events(self) -> None:
        """0.5 pushed these through the text channel as ``[codex stderr]`` prose."""
        parser, events = _parser()
        parser.feed_stderr("panic: index out of bounds\n")
        assert events == [Stderr("panic: index out of bounds")]

    def test_blank_stderr_is_dropped(self) -> None:
        parser, events = _parser()
        parser.feed_stderr("\n")
        assert events == []
