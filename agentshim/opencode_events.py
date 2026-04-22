from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .utils import truncate_params


class OpencodeEvent(ABC):
    """Base class for Opencode stream events."""

    @abstractmethod
    def render(self, log_prefix: str) -> str | None:
        """Render the event as a string for terminal output."""

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

    def render(self, log_prefix: str) -> str | None:
        # We will handle text printing in the agent loop to handle potential streaming
        # or just print it as is.
        # For now, let's return it.
        return self.text


class ToolUseEvent(OpencodeEvent):
    def __init__(self, tool_name: str, input_data: Any, output_data: Any, status: str):
        self.tool_name = tool_name
        self.input_data = input_data
        self.output_data = output_data
        self.status = status

    def render(self, log_prefix: str) -> str:
        # Render tool use and result
        truncated_input = truncate_params(str(self.input_data))

        output_str = ""
        if self.output_data:
            # Truncate output to first 5 lines
            out_lines = str(self.output_data).splitlines()
            truncated_output = "\n".join(out_lines[:5] + (["..."] if len(out_lines) > 5 else []))
            output_str = f"\n{log_prefix} \033[32m[Tool Result] {truncated_output}\033[0m"

        return f"{log_prefix} \033[34m[Tool Use] {self.tool_name} {truncated_input}\033[0m{output_str}"


class StepStartEvent(OpencodeEvent):
    def render(self, log_prefix: str) -> str | None:
        return None


class StepFinishEvent(OpencodeEvent):
    def __init__(self, reason: str | None, cost: float | None, tokens: dict[str, Any] | None):
        self.reason = reason
        self.cost = cost
        self.tokens = tokens

    def render(self, log_prefix: str) -> str | None:
        # Optional: Print cost info?
        # For now, maybe just ignore or print verbose.
        # Let's keep it clean.
        return None
