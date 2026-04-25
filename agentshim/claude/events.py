from __future__ import annotations

from typing import Any, cast


class ClaudeEvent:
    """Base class for Claude Code stream events."""

    @staticmethod
    def from_dict(data: dict[str, Any]) -> ClaudeEvent | None:
        """Factory method to create events from JSON data."""
        event_type = data.get("type")

        if event_type == "system":
            return SystemEvent(data)
        if event_type == "assistant":
            message = data.get("message", {})
            content_blocks = message.get("content", [])
            events: list[ClaudeEvent] = []
            for block in content_blocks:
                block_type = block.get("type")
                if block_type == "text":
                    events.append(TextEvent(block.get("text", "")))
                elif block_type == "tool_use":
                    events.append(
                        ToolUseEvent(
                            tool_name=block.get("name", "Tool"),
                            tool_id=block.get("id"),
                            parameters=block.get("input"),
                        )
                    )
            usage = message.get("usage")
            return MultiEvent(events, usage=usage) if events else None
        if event_type == "user":
            message = data.get("message", {})
            content_blocks = message.get("content", [])
            for block in content_blocks:
                if block.get("type") == "tool_result":
                    return ToolResultEvent(
                        output=block.get("content", ""),
                        tool_id=block.get("tool_use_id"),
                    )
            return None
        if event_type == "result":
            return ResultEvent(
                result=data.get("result", ""),
                num_turns=data.get("num_turns"),
                usage=data.get("usage"),
                total_cost_usd=data.get("total_cost_usd"),
                duration_ms=data.get("duration_ms"),
            )

        return None


class MultiEvent(ClaudeEvent):
    """Container for multiple events from a single message."""

    def __init__(
        self,
        events: list[ClaudeEvent],
        usage: dict[str, Any] | None = None,
    ):
        self.events = events
        self.usage = usage


class SystemEvent(ClaudeEvent):
    """System initialization event.

    Carries the provider ``session_id`` on the ``init`` subtype, used to
    enable conversation resumption via ``claude --resume <id>``.
    """

    def __init__(self, data: dict[str, Any]):
        self.data = data
        self.session_id: str | None = data.get("session_id")


class TextEvent(ClaudeEvent):
    """Assistant text content event."""

    def __init__(self, text: str):
        self.text = text


class ToolUseEvent(ClaudeEvent):
    """Tool call event from assistant."""

    def __init__(self, tool_name: str, tool_id: str | None, parameters: Any):
        self.tool_name = tool_name
        self.tool_id = tool_id
        self.parameters = parameters


class ToolResultEvent(ClaudeEvent):
    """Tool execution result event."""

    def __init__(self, output: Any, tool_id: str | None):
        # Convert output to string if it's not already
        if isinstance(output, list):
            # Handle list content (e.g., from tool_result blocks with multiple items)
            self.output = "\n".join(str(item) for item in cast("list[Any]", output))
        else:
            self.output = str(output) if output else ""
        self.tool_id = tool_id
        self.tool_name_resolved: str = "Tool"  # To be set externally


class ResultEvent(ClaudeEvent):
    """Final session summary event."""

    def __init__(
        self,
        result: str,
        num_turns: int | None = None,
        usage: dict[str, Any] | None = None,
        total_cost_usd: float | None = None,
        duration_ms: int | None = None,
    ):
        self.result = result
        self.num_turns = num_turns
        self.usage = usage
        self.total_cost_usd = total_cost_usd
        self.duration_ms = duration_ms
