from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, cast

from .utils import truncate_content, truncate_params


class ClaudeEvent(ABC):
    """Base class for Claude Code stream events."""

    @abstractmethod
    def render(self, log_prefix: str) -> str | None:
        """Render the event as a string for terminal output."""

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
            return MultiEvent(events) if events else None
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
            return ResultEvent(data.get("result", ""))

        return None


class MultiEvent(ClaudeEvent):
    """Container for multiple events from a single message."""

    def __init__(self, events: list[ClaudeEvent]):
        self.events = events

    def render(self, log_prefix: str) -> str | None:
        # MultiEvent doesn't render itself; events are handled individually
        return None


class SystemEvent(ClaudeEvent):
    """System initialization event."""

    def __init__(self, data: dict[str, Any]):
        self.data = data

    def render(self, log_prefix: str) -> str | None:
        # System events are silent
        return None


class TextEvent(ClaudeEvent):
    """Assistant text content event."""

    def __init__(self, text: str):
        self.text = text

    def render(self, log_prefix: str) -> str | None:
        # Text rendering is handled specially due to streaming
        return self.text


class ToolUseEvent(ClaudeEvent):
    """Tool call event from assistant."""

    def __init__(self, tool_name: str, tool_id: str | None, parameters: Any):
        self.tool_name = tool_name
        self.tool_id = tool_id
        self.parameters = parameters

    def render(self, log_prefix: str) -> str:
        truncated = truncate_params(self.parameters)
        return f"{log_prefix} \033[34m[Tool Use] {self.tool_name} {truncated}\033[0m"


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

    def render(self, log_prefix: str) -> str:
        if not self.output:
            return f"{log_prefix} \033[32m{self.tool_name_resolved} ran successfully\033[0m"
        truncated = truncate_content(self.output)
        return f"{log_prefix} \033[32m[Tool Result] {truncated}\033[0m"


class ResultEvent(ClaudeEvent):
    """Final session summary event."""

    def __init__(self, result: str):
        self.result = result

    def render(self, log_prefix: str) -> str | None:
        # Result events are silent (result is captured separately)
        return None
