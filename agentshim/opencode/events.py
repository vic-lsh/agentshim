from __future__ import annotations

from typing import Any


class OpencodeEvent:
    """Base class for Opencode stream events."""

    @staticmethod
    def from_dict(data: dict[str, Any]) -> OpencodeEvent | None:
        """Factory method to create events from JSON data."""
        msg_type = data.get("type")
        part = data.get("part", {})

        if msg_type == "text":
            return TextEvent(text=part.get("text", ""))
        if msg_type == "tool_use":
            return ToolUseEvent(
                tool_name=part.get("tool", "Tool"),
                input_data=part.get("state", {}).get("input"),
                output_data=part.get("state", {}).get("output"),
                status=part.get("state", {}).get("status"),
            )
        if msg_type == "step_start":
            return StepStartEvent()
        if msg_type == "step_finish":
            return StepFinishEvent(
                reason=part.get("reason"),
                cost=part.get("cost"),
                tokens=part.get("tokens"),
            )

        return None


class TextEvent(OpencodeEvent):
    def __init__(self, text: str):
        self.text = text


class ToolUseEvent(OpencodeEvent):
    def __init__(self, tool_name: str, input_data: Any, output_data: Any, status: str):
        self.tool_name = tool_name
        self.input_data = input_data
        self.output_data = output_data
        self.status = status


class StepStartEvent(OpencodeEvent):
    pass


class StepFinishEvent(OpencodeEvent):
    def __init__(self, reason: str | None, cost: float | None, tokens: dict[str, Any] | None):
        self.reason = reason
        self.cost = cost
        self.tokens = tokens
