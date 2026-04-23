from agentshim.claude.events import (
    ClaudeEvent,
    MultiEvent,
    ResultEvent,
    SystemEvent,
    TextEvent,
    ToolResultEvent,
    ToolUseEvent,
)
from agentshim.copilot.events import CopilotEvent, MessageDeltaEvent as CopilotMessageDeltaEvent
from agentshim.copilot.events import MessageEvent as CopilotMessageEvent
from agentshim.copilot.events import ResultEvent as CopilotResultEvent
from agentshim.copilot.events import ToolResultEvent as CopilotToolResultEvent
from agentshim.copilot.events import ToolUseEvent as CopilotToolUseEvent


class TestClaudeEventFromDict:
    """Tests for ClaudeEvent.from_dict factory method."""

    def test_system_event(self):
        data = {"type": "system", "subtype": "init", "session_id": "abc"}
        event = ClaudeEvent.from_dict(data)
        assert isinstance(event, SystemEvent)

    def test_assistant_text_event(self):
        data = {
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": "Hello world"}]},
        }
        event = ClaudeEvent.from_dict(data)
        assert isinstance(event, MultiEvent)
        assert len(event.events) == 1
        assert isinstance(event.events[0], TextEvent)
        assert event.events[0].text == "Hello world"

    def test_assistant_tool_use_event(self):
        data = {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "name": "Bash",
                        "id": "tool_1",
                        "input": {"command": "ls -la"},
                    }
                ]
            },
        }
        event = ClaudeEvent.from_dict(data)
        assert isinstance(event, MultiEvent)
        assert len(event.events) == 1
        tool_event = event.events[0]
        assert isinstance(tool_event, ToolUseEvent)
        assert tool_event.tool_name == "Bash"
        assert tool_event.tool_id == "tool_1"
        assert tool_event.parameters == {"command": "ls -la"}

    def test_assistant_multiple_content_blocks(self):
        data = {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "text", "text": "Let me check."},
                    {
                        "type": "tool_use",
                        "name": "Read",
                        "id": "t1",
                        "input": {"path": "/tmp/f.py"},
                    },
                ]
            },
        }
        event = ClaudeEvent.from_dict(data)
        assert isinstance(event, MultiEvent)
        assert len(event.events) == 2
        assert isinstance(event.events[0], TextEvent)
        assert isinstance(event.events[1], ToolUseEvent)

    def test_assistant_event_captures_usage(self):
        data = {
            "type": "assistant",
            "message": {
                "content": [{"type": "text", "text": "hi"}],
                "usage": {
                    "input_tokens": 1234,
                    "output_tokens": 56,
                    "cache_creation_input_tokens": 100,
                    "cache_read_input_tokens": 7,
                },
            },
        }
        event = ClaudeEvent.from_dict(data)
        assert isinstance(event, MultiEvent)
        assert event.usage == {
            "input_tokens": 1234,
            "output_tokens": 56,
            "cache_creation_input_tokens": 100,
            "cache_read_input_tokens": 7,
        }

    def test_assistant_event_missing_usage_is_none(self):
        data = {
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": "hi"}]},
        }
        event = ClaudeEvent.from_dict(data)
        assert isinstance(event, MultiEvent)
        assert event.usage is None

    def test_user_tool_result_event(self):
        data = {
            "type": "user",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": "file contents here",
                    }
                ]
            },
        }
        event = ClaudeEvent.from_dict(data)
        assert isinstance(event, ToolResultEvent)
        assert event.output == "file contents here"
        assert event.tool_id == "t1"

    def test_user_message_without_tool_result_returns_none(self):
        data = {
            "type": "user",
            "message": {"content": [{"type": "text", "text": "user says something"}]},
        }
        event = ClaudeEvent.from_dict(data)
        assert event is None

    def test_result_event(self):
        data = {"type": "result", "result": "Task completed successfully."}
        event = ClaudeEvent.from_dict(data)
        assert isinstance(event, ResultEvent)
        assert event.result == "Task completed successfully."
        assert event.num_turns is None
        assert event.usage is None
        assert event.total_cost_usd is None

    def test_result_event_carries_usage_and_turns(self):
        data = {
            "type": "result",
            "result": "done",
            "num_turns": 7,
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_creation_input_tokens": 30,
                "cache_read_input_tokens": 20,
            },
            "total_cost_usd": 0.0123,
        }
        event = ClaudeEvent.from_dict(data)
        assert isinstance(event, ResultEvent)
        assert event.num_turns == 7
        assert event.usage is not None
        assert event.usage["input_tokens"] == 100
        assert event.usage["cache_read_input_tokens"] == 20
        assert event.total_cost_usd == 0.0123

    def test_unknown_event_type_returns_none(self):
        data = {"type": "unknown_custom_type", "data": {}}
        event = ClaudeEvent.from_dict(data)
        assert event is None

    def test_missing_type_returns_none(self):
        data = {"data": "no type field"}
        event = ClaudeEvent.from_dict(data)
        assert event is None

    def test_assistant_empty_content_returns_none(self):
        data = {"type": "assistant", "message": {"content": []}}
        event = ClaudeEvent.from_dict(data)
        assert event is None

    def test_assistant_missing_message_returns_none(self):
        data = {"type": "assistant"}
        event = ClaudeEvent.from_dict(data)
        assert event is None


class TestToolResultEventOutputParsing:
    """Tests for ToolResultEvent output handling."""

    def test_string_output(self):
        event = ToolResultEvent(output="simple string", tool_id="t1")
        assert event.output == "simple string"

    def test_list_output_joined(self):
        event = ToolResultEvent(output=["line1", "line2", "line3"], tool_id="t1")
        assert event.output == "line1\nline2\nline3"

    def test_none_output_becomes_empty_string(self):
        event = ToolResultEvent(output=None, tool_id="t1")
        assert event.output == ""

    def test_numeric_output_becomes_string(self):
        event = ToolResultEvent(output=42, tool_id="t1")
        assert event.output == "42"


class TestEventRendering:
    """Tests for event render methods."""

    def test_system_event_renders_none(self):
        event = SystemEvent({"type": "system"})
        assert event.render("[P]") is None

    def test_text_event_renders_text(self):
        event = TextEvent("hello")
        assert event.render("[P]") == "hello"

    def test_tool_use_event_renders_with_prefix(self):
        event = ToolUseEvent("Bash", "t1", {"cmd": "ls"})
        rendered = event.render("[Claude]")
        assert "[Claude]" in rendered
        assert "Bash" in rendered
        assert "[Tool Use]" in rendered

    def test_memory_tool_use_event_preserves_large_payload(self):
        event = ToolUseEvent("store_incident", "t1", {"summary": "x" * 300})
        rendered = event.render("[Claude]")
        assert "[Tool Use]" in rendered
        assert "..." not in rendered
        assert "x" * 300 in rendered

    def test_tool_result_event_renders_with_output(self):
        event = ToolResultEvent(output="file.txt", tool_id="t1")
        event.tool_name_resolved = "Bash"
        rendered = event.render("[Claude]")
        assert "[Tool Result]" in rendered
        assert "file.txt" in rendered

    def test_tool_result_event_renders_success_when_empty(self):
        event = ToolResultEvent(output="", tool_id="t1")
        event.tool_name_resolved = "Bash"
        rendered = event.render("[Claude]")
        assert "ran successfully" in rendered

    def test_result_event_renders_none(self):
        event = ResultEvent("done")
        assert event.render("[P]") is None

    def test_multi_event_renders_none(self):
        event = MultiEvent([TextEvent("a")])
        assert event.render("[P]") is None


class TestCopilotEventFromDict:
    def test_assistant_message_event(self):
        event = CopilotEvent.from_dict(
            {"type": "assistant.message", "data": {"messageId": "m1", "content": "Hello", "outputTokens": 5}}
        )
        assert isinstance(event, CopilotMessageEvent)
        assert event.message_id == "m1"
        assert event.content == "Hello"
        assert event.output_tokens == 5

    def test_assistant_message_delta_event(self):
        event = CopilotEvent.from_dict(
            {"type": "assistant.message_delta", "ephemeral": True, "data": {"messageId": "m1", "deltaContent": "Hi"}}
        )
        assert isinstance(event, CopilotMessageDeltaEvent)
        assert event.message_id == "m1"
        assert event.delta_content == "Hi"

    def test_tool_execution_start_event(self):
        event = CopilotEvent.from_dict(
            {
                "type": "tool.execution_start",
                "data": {"toolCallId": "t1", "toolName": "shell", "arguments": {"command": "ls"}},
            }
        )
        assert isinstance(event, CopilotToolUseEvent)
        assert event.tool_id == "t1"
        assert event.tool_name == "shell"
        assert event.arguments == {"command": "ls"}

    def test_tool_execution_complete_uses_detailed_content_and_exit_code(self):
        event = CopilotEvent.from_dict(
            {
                "type": "tool.execution_complete",
                "data": {
                    "toolCallId": "t1",
                    "success": True,
                    "result": {
                        "content": "short",
                        "detailedContent": "full output",
                        "contents": [{"type": "terminal", "text": "full output", "exitCode": 7}],
                    },
                },
            }
        )
        assert isinstance(event, CopilotToolResultEvent)
        assert event.tool_id == "t1"
        assert event.output == "full output"
        assert event.exit_code == 7

    def test_result_event(self):
        event = CopilotEvent.from_dict({"type": "result", "sessionId": "sid-1", "exitCode": 0, "usage": {}})
        assert isinstance(event, CopilotResultEvent)
        assert event.session_id == "sid-1"
        assert event.exit_code == 0

    def test_unknown_event_returns_none(self):
        assert CopilotEvent.from_dict({"type": "session.tools_updated", "data": {"model": "gpt-4.1"}}) is None


class TestCodexEventFromDict:
    """Tests for CodexEvent.from_dict factory method."""

    def test_thread_started_carries_thread_id(self):
        from agentshim.codex_events import CodexEvent, ThreadStartedEvent

        event = CodexEvent.from_dict({"type": "thread.started", "thread_id": "abc"})
        assert isinstance(event, ThreadStartedEvent)
        assert event.thread_id == "abc"
        assert event.render("[Codex]") is None

    def test_turn_started_is_lifecycle(self):
        from agentshim.codex_events import CodexEvent, LifecycleEvent

        event = CodexEvent.from_dict({"type": "turn.started"})
        assert isinstance(event, LifecycleEvent)

    def test_turn_completed_carries_usage(self):
        from agentshim.codex_events import CodexEvent, TurnCompletedEvent

        event = CodexEvent.from_dict(
            {
                "type": "turn.completed",
                "usage": {
                    "input_tokens": 10,
                    "cached_input_tokens": 3,
                    "output_tokens": 4,
                },
            }
        )
        assert isinstance(event, TurnCompletedEvent)
        assert event.input_tokens == 10
        assert event.cached_input_tokens == 3
        assert event.output_tokens == 4

    def test_agent_message_completed_is_text(self):
        from agentshim.codex_events import CodexEvent
        from agentshim.codex_events import TextEvent as CodexTextEvent

        data = {
            "type": "item.completed",
            "item": {"id": "item_0", "type": "agent_message", "text": "Hello"},
        }
        event = CodexEvent.from_dict(data)
        assert isinstance(event, CodexTextEvent)
        assert event.text == "Hello"

    def test_agent_message_started_is_skipped(self):
        from agentshim.codex_events import CodexEvent

        data = {
            "type": "item.started",
            "item": {"id": "item_0", "type": "agent_message", "text": ""},
        }
        assert CodexEvent.from_dict(data) is None

    def test_command_execution_started_is_tool_use(self):
        from agentshim.codex_events import CodexEvent
        from agentshim.codex_events import ToolUseEvent as CodexToolUseEvent

        data = {
            "type": "item.started",
            "item": {
                "id": "item_1",
                "type": "command_execution",
                "command": "/bin/bash -lc ls",
                "status": "in_progress",
            },
        }
        event = CodexEvent.from_dict(data)
        assert isinstance(event, CodexToolUseEvent)
        assert event.tool_id == "item_1"
        assert event.tool_name == "shell"
        assert event.parameters == {"command": "/bin/bash -lc ls"}

    def test_command_execution_completed_is_tool_result(self):
        from agentshim.codex_events import CodexEvent
        from agentshim.codex_events import ToolResultEvent as CodexToolResultEvent

        data = {
            "type": "item.completed",
            "item": {
                "id": "item_1",
                "type": "command_execution",
                "command": "/bin/bash -lc ls",
                "aggregated_output": "file.txt\n",
                "exit_code": 0,
                "status": "completed",
            },
        }
        event = CodexEvent.from_dict(data)
        assert isinstance(event, CodexToolResultEvent)
        assert event.tool_id == "item_1"
        assert event.output == "file.txt\n"
        assert event.exit_code == 0
        assert event.status == "completed"

    def test_generic_item_types_become_tool_events(self):
        from agentshim.codex_events import CodexEvent
        from agentshim.codex_events import ToolUseEvent as CodexToolUseEvent

        data = {
            "type": "item.started",
            "item": {"id": "r1", "type": "reasoning", "summary": "planning"},
        }
        event = CodexEvent.from_dict(data)
        assert isinstance(event, CodexToolUseEvent)
        assert event.tool_name == "reasoning"

    def test_turn_failed_is_error(self):
        from agentshim.codex_events import CodexEvent, ErrorEvent

        data = {"type": "turn.failed", "error": {"message": "boom"}}
        event = CodexEvent.from_dict(data)
        assert isinstance(event, ErrorEvent)
        assert event.message == "boom"

    def test_top_level_error_event(self):
        from agentshim.codex_events import CodexEvent, ErrorEvent

        event = CodexEvent.from_dict({"type": "error", "message": "bad"})
        assert isinstance(event, ErrorEvent)
        assert event.message == "bad"

    def test_unknown_event_returns_none(self):
        from agentshim.codex_events import CodexEvent

        assert CodexEvent.from_dict({"type": "mystery"}) is None
