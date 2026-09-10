"""Copilot CLI ``--output-format json`` parsing."""

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
from agentshim.providers.copilot import CopilotStreamParser


def _parser() -> tuple[CopilotStreamParser, list[AgentEvent]]:
    events: list[AgentEvent] = []
    return CopilotStreamParser(events.append), events


def _line(payload: dict[str, Any]) -> str:
    return json.dumps(payload) + "\n"


def _frame(kind: str, **data: object) -> str:
    return _line({"type": kind, "data": data})


def _usage_frame(**counts: object) -> str:
    return _frame("assistant.usage", model="gpt-5", **counts)


class TestSessionId:
    def test_a_session_start_frame_starts_the_session(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(_frame("session.start", sessionId="abc-123"))
        assert events == [SessionStarted("abc-123")]
        assert parser.finish().session_id == "abc-123"

    def test_the_result_frame_names_the_session_when_nothing_else_did(self) -> None:
        """Copilot 1.0.x only reports the session id in its terminal frame."""
        parser, events = _parser()
        parser.feed_stdout(_line({"type": "result", "sessionId": "res-1", "exitCode": 0}))
        assert SessionStarted("res-1") in events
        assert parser.finish().session_id == "res-1"

    def test_only_the_first_id_is_kept(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(_frame("session.start", sessionId="first"))
        parser.feed_stdout(_line({"type": "result", "sessionId": "second", "exitCode": 0}))
        assert [event for event in events if isinstance(event, SessionStarted)] == [
            SessionStarted("first")
        ]
        assert parser.finish().session_id == "first"

    def test_a_frame_without_an_id_emits_nothing(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(_frame("session.start"))
        assert events == []
        assert parser.finish().session_id is None


class TestAssistantText:
    def test_a_complete_message_emits_and_becomes_the_final_text(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(_frame("assistant.message", messageId="m1", content="Hello"))
        assert events == [AssistantText("Hello")]
        assert parser.finish().text == "Hello"

    def test_deltas_stream_as_they_arrive(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(_frame("assistant.message_delta", messageId="m1", deltaContent="Hel"))
        parser.feed_stdout(_frame("assistant.message_delta", messageId="m1", deltaContent="lo"))
        assert events == [AssistantText("Hel"), AssistantText("lo")]

    def test_a_streamed_message_is_not_emitted_twice(self) -> None:
        """The complete frame only confirms what the deltas already carried."""
        parser, events = _parser()
        parser.feed_stdout(_frame("assistant.message_delta", messageId="m1", deltaContent="Hello"))
        parser.feed_stdout(_frame("assistant.message", messageId="m1", content="Hello"))
        assert events == [AssistantText("Hello")]
        assert parser.finish().text == "Hello"

    def test_deltas_are_the_text_when_no_complete_message_arrives(self) -> None:
        parser, _ = _parser()
        parser.feed_stdout(_frame("assistant.message_delta", messageId="m1", deltaContent=" Hel"))
        parser.feed_stdout(_frame("assistant.message_delta", messageId="m1", deltaContent="lo "))
        assert parser.finish().text == "Hello"

    def test_a_message_of_another_id_is_still_emitted(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(_frame("assistant.message_delta", messageId="m1", deltaContent="Hi"))
        parser.feed_stdout(_frame("assistant.message", messageId="m2", content="Bye"))
        assert events == [AssistantText("Hi"), AssistantText("Bye")]

    def test_the_last_complete_message_wins(self) -> None:
        parser, _ = _parser()
        parser.feed_stdout(_frame("assistant.message", messageId="m1", content="first"))
        parser.feed_stdout(_frame("assistant.message", messageId="m2", content="second"))
        assert parser.finish().text == "second"

    def test_an_empty_message_is_ignored(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(_frame("assistant.message", messageId="m1", content=""))
        assert events == []
        assert parser.finish().text == ""

    def test_an_empty_stream_yields_empty_text(self) -> None:
        parser, _ = _parser()
        assert parser.finish().text == ""


class TestReasoning:
    def test_an_intent_becomes_reasoning(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(_frame("assistant.intent", intent="reading the config"))
        assert events == [Reasoning("reading the config")]
        assert parser.finish().text == ""

    def test_an_intent_without_text_is_ignored(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(_frame("assistant.intent"))
        assert events == []


class TestTools:
    def test_a_call_and_its_result_are_paired(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(
            _frame(
                "tool.execution_start", toolCallId="t1", toolName="shell", arguments={"cmd": "ls"}
            )
        )
        parser.feed_stdout(
            _frame(
                "tool.execution_complete",
                toolCallId="t1",
                success=True,
                result={"content": "file.txt"},
            )
        )
        assert events[0] == ToolCall("t1", "shell", {"cmd": "ls"})
        result = events[-1]
        assert isinstance(result, ToolResult)
        assert result.tool == "shell"
        assert result.stdout == "file.txt"
        assert result.stderr == ""
        assert result.exit_code is None
        assert result.duration_s is not None
        assert result.duration_s >= 0

    def test_an_unpaired_result_has_no_duration_and_a_fallback_name(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(_frame("tool.execution_complete", toolCallId="t9", success=True))
        result = events[-1]
        assert isinstance(result, ToolResult)
        assert result.tool == "Tool"
        assert result.duration_s is None

    def test_an_unnamed_tool_falls_back(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(_frame("tool.execution_start", toolCallId="t1"))
        call = events[0]
        assert isinstance(call, ToolCall)
        assert call.tool == "Tool"
        assert call.args is None

    def test_detailed_content_wins_over_the_abbreviated_form(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(
            _frame(
                "tool.execution_complete",
                toolCallId="t1",
                success=True,
                result={"content": "ok", "detailedContent": "ok\nfull"},
            )
        )
        result = events[-1]
        assert isinstance(result, ToolResult)
        assert result.stdout == "ok\nfull"

    def test_a_terminal_block_supplies_the_exit_code(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(
            _frame(
                "tool.execution_complete",
                toolCallId="t1",
                success=True,
                result={"contents": [{"type": "terminal", "text": "out", "exitCode": 3}]},
            )
        )
        result = events[-1]
        assert isinstance(result, ToolResult)
        assert result.stdout == "out"
        assert result.exit_code == 3

    def test_resource_links_are_rendered(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(
            _frame(
                "tool.execution_complete",
                toolCallId="t1",
                success=True,
                result={
                    "contents": [
                        {"type": "resource_link", "title": "spec", "uri": "file:///spec.md"}
                    ]
                },
            )
        )
        result = events[-1]
        assert isinstance(result, ToolResult)
        assert result.stdout == "spec: file:///spec.md"

    def test_a_failed_tool_lands_on_stderr(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(
            _frame(
                "tool.execution_complete",
                toolCallId="t1",
                success=False,
                error={"message": "permission denied"},
            )
        )
        result = events[-1]
        assert isinstance(result, ToolResult)
        assert result.stdout == ""
        assert result.stderr == "permission denied"
        assert result.exit_code == 1

    def test_a_failed_tool_keeps_the_exit_code_it_reported(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(
            _frame(
                "tool.execution_complete",
                toolCallId="t1",
                success=False,
                result={"contents": [{"type": "terminal", "text": "boom", "exitCode": 127}]},
            )
        )
        result = events[-1]
        assert isinstance(result, ToolResult)
        assert result.stderr == "boom"
        assert result.exit_code == 127


class TestUsage:
    def test_cache_and_reasoning_tokens_fold_in(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(
            _usage_frame(
                inputTokens=100,
                outputTokens=20,
                cacheReadTokens=10,
                cacheWriteTokens=5,
                reasoningTokens=4,
            )
        )
        parser.feed_stdout(_frame("assistant.turn_end", turnId="0"))
        tokens = parser.finish().usage.tokens
        assert tokens.input_tokens == 115
        assert tokens.output_tokens == 24
        assert tokens.cached_input_tokens == 15
        assert tokens.cache_write_input_tokens == 5
        assert tokens.reasoning_output_tokens == 4
        assert tokens.turns == 1
        assert tokens.cached_input_tokens <= tokens.input_tokens
        assert isinstance(events[0], UsageReport)

    def test_usage_frames_accumulate(self) -> None:
        parser, _ = _parser()
        parser.feed_stdout(_usage_frame(inputTokens=10, outputTokens=1))
        parser.feed_stdout(_usage_frame(inputTokens=20, outputTokens=2))
        tokens = parser.finish().usage.tokens
        assert tokens.input_tokens == 30
        assert tokens.output_tokens == 3

    def test_each_usage_frame_reports_only_its_own_counts(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(_usage_frame(inputTokens=10, outputTokens=1))
        parser.feed_stdout(_usage_frame(inputTokens=20, outputTokens=2))
        reports = [event for event in events if isinstance(event, UsageReport)]
        assert [report.usage.tokens.input_tokens for report in reports] == [10, 20]
        assert all(report.cost_usd is None for report in reports)

    def test_message_output_tokens_are_the_fallback(self) -> None:
        """Copilot 1.0.x prints no usage frame; the message count is all there is."""
        parser, _ = _parser()
        parser.feed_stdout(
            _frame("assistant.message", messageId="m1", content="hi", outputTokens=5)
        )
        parser.feed_stdout(_frame("assistant.turn_end", turnId="0"))
        tokens = parser.finish().usage.tokens
        assert tokens.output_tokens == 5
        assert tokens.input_tokens == 0
        assert tokens.turns == 1

    def test_a_usage_frame_beats_the_message_fallback(self) -> None:
        parser, _ = _parser()
        parser.feed_stdout(
            _frame("assistant.message", messageId="m1", content="hi", outputTokens=5)
        )
        parser.feed_stdout(_usage_frame(inputTokens=100, outputTokens=20))
        assert parser.finish().usage.tokens.output_tokens == 20

    def test_the_raw_mapping_is_preserved(self) -> None:
        parser, _ = _parser()
        parser.feed_stdout(_usage_frame(inputTokens=7000, outputTokens=120))
        raw = parser.finish().usage.raw
        assert raw is not None
        assert raw["inputTokens"] == 7000

    def test_session_totals_are_the_raw_fallback(self) -> None:
        parser, _ = _parser()
        parser.feed_stdout(
            _line({"type": "result", "sessionId": "s", "usage": {"premiumRequests": 2}})
        )
        assert parser.finish().usage.raw == {"premiumRequests": 2}

    def test_the_terminal_frame_always_reports_usage(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(_frame("assistant.turn_end", turnId="0"))
        parser.feed_stdout(_line({"type": "result", "sessionId": "s", "exitCode": 0}))
        report = events[-1]
        assert isinstance(report, UsageReport)
        assert report.usage.provider == "copilot"
        assert report.usage.tokens.turns == 1

    def test_no_usage_at_all_still_yields_a_copilot_provider_usage(self) -> None:
        parser, _ = _parser()
        parsed = parser.finish()
        assert parsed.usage.provider == "copilot"
        assert parsed.usage.raw is None
        assert parsed.cost_usd is None


class TestErrors:
    def test_a_session_error_emits_a_provider_error(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(_frame("session.error", errorType="rate_limit", message="slow down"))
        assert ProviderError("slow down") in events
        assert parser.finish().error == "slow down"

    def test_the_error_type_is_used_when_there_is_no_message(self) -> None:
        parser, _ = _parser()
        parser.feed_stdout(_frame("session.error", errorType="rate_limit"))
        assert parser.finish().error == "rate_limit"

    def test_a_clean_run_reports_no_error(self) -> None:
        parser, _ = _parser()
        parser.feed_stdout(_frame("assistant.message", messageId="m1", content="ok"))
        assert parser.finish().error is None


class TestStructuredOutput:
    def test_copilot_never_produces_structured_output(self) -> None:
        parser, _ = _parser()
        parser.feed_stdout(_frame("assistant.message", messageId="m1", content='{"a": 3}'))
        assert parser.finish().structured_output is None


class TestMalformedLines:
    def test_a_non_json_line_becomes_raw_output(self) -> None:
        parser, events = _parser()
        parser.feed_stdout("starting copilot...\n")
        assert events == [RawOutput("starting copilot...")]

    def test_a_non_object_json_line_does_not_crash(self) -> None:
        """A bare scalar or array on stdout must not blow up ``data.get``."""
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
        parser.feed_stdout(_frame("session.mcp_servers_loaded", servers=[]))
        assert events == []

    def test_a_frame_without_a_type_is_ignored(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(_line({"sessionId": "x"}))
        assert events == []

    def test_a_frame_whose_data_is_not_an_object_is_tolerated(self) -> None:
        parser, events = _parser()
        parser.feed_stdout(_line({"type": "assistant.message", "data": "nope"}))
        assert events == []

    def test_junk_token_counts_degrade_to_zero(self) -> None:
        parser, _ = _parser()
        parser.feed_stdout(_usage_frame(inputTokens="lots", outputTokens=None))
        assert parser.finish().usage.tokens.input_tokens == 0


class TestStderr:
    def test_stderr_lines_are_emitted(self) -> None:
        parser, events = _parser()
        parser.feed_stderr("warning: something\n")
        assert events == [Stderr("warning: something")]

    def test_blank_stderr_is_dropped(self) -> None:
        parser, events = _parser()
        parser.feed_stderr("\n")
        assert events == []
