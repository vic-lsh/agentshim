from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, cast

from ..utils import truncate_content, truncate_tool_params


class CodexEvent(ABC):
    """Base class for Codex (``codex exec --json``) stream events."""

    @abstractmethod
    def render(self, log_prefix: str) -> str | None:
        """Render the event as a string for terminal output."""

    @staticmethod
    def from_dict(data: dict[str, Any]) -> CodexEvent | None:
        """Factory method to create events from JSON data."""
        event_type = data.get("type")

        if event_type == "thread.started":
            thread_id_raw = data.get("thread_id")
            thread_id = thread_id_raw if isinstance(thread_id_raw, str) else None
            return ThreadStartedEvent(thread_id=thread_id)

        if event_type == "turn.started":
            return LifecycleEvent(event_type)

        if event_type == "turn.completed":
            return TurnCompletedEvent.from_usage_payload(data.get("usage"))

        if event_type in ("item.started", "item.completed"):
            item_raw = data.get("item")
            if not isinstance(item_raw, dict):
                return None
            item = cast("dict[str, Any]", item_raw)
            item_type_raw = item.get("type")
            item_type = item_type_raw if isinstance(item_type_raw, str) else None
            item_id_raw = item.get("id")
            item_id = item_id_raw if isinstance(item_id_raw, str) else None
            completed = event_type == "item.completed"

            if item_type == "agent_message":
                if not completed:
                    return None
                text_raw = item.get("text", "")
                return TextEvent(text=text_raw if isinstance(text_raw, str) else "")

            if item_type == "reasoning":
                if not completed:
                    return None
                text_raw = item.get("text", "")
                return TextEvent(text=text_raw if isinstance(text_raw, str) else "")

            if item_type == "command_execution":
                command_raw = item.get("command", "")
                command = command_raw if isinstance(command_raw, str) else ""
                exit_code_raw = item.get("exit_code")
                exit_code = exit_code_raw if isinstance(exit_code_raw, int) else None
                status_raw = item.get("status")
                status = status_raw if isinstance(status_raw, str) else None
                if completed:
                    return ToolResultEvent(
                        tool_id=item_id,
                        output=item.get("aggregated_output", ""),
                        exit_code=exit_code,
                        status=status,
                        tool_name="execute",
                        parameters={"command": command},
                    )
                return ToolUseEvent(
                    tool_id=item_id,
                    tool_name="execute",
                    parameters={"command": command},
                )

            status_raw = item.get("status")
            status = status_raw if isinstance(status_raw, str) else None
            if completed:
                return ToolUseEvent(
                    tool_id=item_id,
                    tool_name=item_type or "item",
                    parameters=_item_parameters(item),
                )
            return ToolUseEvent(
                tool_id=item_id,
                tool_name=item_type or "item",
                parameters=_item_parameters(item),
            )

        if event_type == "turn.failed":
            err_raw = data.get("error")
            if isinstance(err_raw, dict):
                err = cast("dict[str, Any]", err_raw)
                msg_raw = err.get("message", "")
                message = msg_raw if isinstance(msg_raw, str) else str(msg_raw)
            else:
                message = str(err_raw) if err_raw else ""
            return ErrorEvent(message=message)

        if event_type == "error":
            message_raw = data.get("message", "")
            return ErrorEvent(message=message_raw if isinstance(message_raw, str) else str(message_raw))

        return None


def _item_parameters(item: dict[str, Any]) -> dict[str, Any]:
    """Extract a parameter dict from a generic codex item payload."""
    excluded = {"id", "type", "status"}
    return {k: v for k, v in item.items() if k not in excluded}


def _summarize_item(item: dict[str, Any]) -> str:
    """Summarize a generic codex item for the tool-result output field."""
    for key in ("text", "summary", "output", "result"):
        value = item.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


class ThreadStartedEvent(CodexEvent):
    """Thread-level start event carrying the resumable ``thread_id``."""

    def __init__(self, thread_id: str | None):
        self.thread_id = thread_id

    def render(self, log_prefix: str) -> str | None:
        return None


class LifecycleEvent(CodexEvent):
    """Turn lifecycle marker; not rendered."""

    def __init__(self, event_type: str):
        self.event_type = event_type

    def render(self, log_prefix: str) -> str | None:
        return None


class TextEvent(CodexEvent):
    """Assistant text content event (``agent_message`` item)."""

    def __init__(self, text: str):
        self.text = text

    def render(self, log_prefix: str) -> str | None:
        return self.text


class ToolUseEvent(CodexEvent):
    """Tool call start event (``item.started``)."""

    def __init__(self, tool_name: str, tool_id: str | None, parameters: Any):
        self.tool_name = tool_name
        self.tool_id = tool_id
        self.parameters = parameters

    def render(self, log_prefix: str) -> str:
        truncated = truncate_tool_params(self.tool_name, self.parameters)
        return f"{log_prefix} \033[34m[Tool Use] {self.tool_name} {truncated}\033[0m"


class ToolResultEvent(CodexEvent):
    """Tool call completion event (``item.completed``)."""

    def __init__(
        self,
        output: Any,
        tool_id: str | None,
        exit_code: int | None = None,
        status: str | None = None,
        tool_name: str | None = None,
        parameters: Any = None,
    ):
        if isinstance(output, list):
            items = cast("list[Any]", output)
            self.output = "\n".join(str(item) for item in items)
        else:
            self.output = str(output) if output else ""
        self.tool_id = tool_id
        self.exit_code = exit_code
        self.status = status
        self.tool_name = tool_name
        self.parameters = parameters
        self.tool_name_resolved: str = tool_name or "Tool"

    def render(self, log_prefix: str) -> str:
        if not self.output:
            return f"{log_prefix} \033[32m{self.tool_name_resolved} ran successfully\033[0m"
        truncated = truncate_content(self.output)
        return f"{log_prefix} \033[32m[Tool Result] {truncated}\033[0m"


class TurnCompletedEvent(CodexEvent):
    """Per-turn usage summary emitted by Codex."""

    def __init__(
        self,
        input_tokens: int = 0,
        cached_input_tokens: int = 0,
        output_tokens: int = 0,
        usage: dict[str, Any] | None = None,
    ):
        self.input_tokens = input_tokens
        self.cached_input_tokens = cached_input_tokens
        self.output_tokens = output_tokens
        self.usage = usage

    @classmethod
    def from_usage_payload(cls, usage_raw: Any) -> TurnCompletedEvent:
        usage = cast("dict[str, Any]", usage_raw) if isinstance(usage_raw, dict) else None
        usage_dict = usage or {}
        return cls(
            input_tokens=int(usage_dict.get("input_tokens") or 0),
            cached_input_tokens=int(usage_dict.get("cached_input_tokens") or 0),
            output_tokens=int(usage_dict.get("output_tokens") or 0),
            usage=usage,
        )

    @property
    def has_usage(self) -> bool:
        return self.usage is not None

    def render(self, log_prefix: str) -> str | None:
        return None


class ErrorEvent(CodexEvent):
    """Error event emitted on turn failure or top-level error."""

    def __init__(self, message: str):
        self.message = message

    def render(self, log_prefix: str) -> str:
        return f"{log_prefix} \033[31m[Error] {self.message}\033[0m"
