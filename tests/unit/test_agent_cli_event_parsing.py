from agentshim.claude_events import (
    ClaudeEvent,
    MultiEvent,
    ResultEvent,
    SystemEvent,
    TextEvent,
    ToolResultEvent,
    ToolUseEvent,
)


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
