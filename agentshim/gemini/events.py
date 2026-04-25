from __future__ import annotations

from typing import Any


class GeminiEvent:
    """Base class for Gemini stream events."""

    @staticmethod
    def from_dict(data: dict[str, Any]) -> GeminiEvent | None:
        """Factory method to create events from JSON data."""
        msg_type = data.get("type")

        if msg_type == "init":
            return InitEvent(session_id=data.get("session_id"))
        if msg_type == "message":
            return MessageEvent(role=data.get("role", ""), content=data.get("content", ""))
        if msg_type == "tool_use":
            return ToolUseEvent(
                tool_name=data.get("tool_name", "Tool"),
                tool_id=data.get("tool_id"),
                parameters=data.get("parameters"),
            )
        if msg_type == "tool_result":
            return ToolResultEvent(output=data.get("output", ""), tool_id=data.get("tool_id"))
        return None


class InitEvent(GeminiEvent):
    """Session initialization event; carries the resumable ``session_id``."""

    def __init__(self, session_id: str | None):
        self.session_id = session_id


class MessageEvent(GeminiEvent):
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content


class ToolUseEvent(GeminiEvent):
    def __init__(self, tool_name: str, tool_id: str | None, parameters: Any):
        self.tool_name = tool_name
        self.tool_id = tool_id
        self.parameters = parameters


class ToolResultEvent(GeminiEvent):
    def __init__(self, output: str, tool_id: str | None):
        self.output = output
        self.tool_id = tool_id
        self.tool_name_resolved: str = "Tool"  # To be set externally
