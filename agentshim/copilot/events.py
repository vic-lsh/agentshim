from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, cast

from ..utils import truncate_content, truncate_tool_params


def _as_dict(value: Any) -> dict[str, Any]:
    return cast("dict[str, Any]", value) if isinstance(value, dict) else {}


def _extract_tool_result(result_raw: Any) -> tuple[str, int | None]:
    result = _as_dict(result_raw)
    exit_code: int | None = None

    detailed = result.get("detailedContent")
    if isinstance(detailed, str) and detailed:
        output = detailed
    else:
        content = result.get("content")
        output = content if isinstance(content, str) else ""

    contents = result.get("contents")
    if isinstance(contents, list):
        rendered_blocks: list[str] = []
        for item_raw in contents:
            item = _as_dict(item_raw)
            item_type = item.get("type")
            if item_type in ("text", "terminal"):
                text = item.get("text")
                if isinstance(text, str) and text:
                    rendered_blocks.append(text)
                if item_type == "terminal":
                    candidate_exit_code = item.get("exitCode")
                    if isinstance(candidate_exit_code, int):
                        exit_code = candidate_exit_code
            elif item_type == "resource_link":
                name = item.get("title") or item.get("name") or item.get("uri")
                uri = item.get("uri")
                if isinstance(name, str) and isinstance(uri, str):
                    rendered_blocks.append(f"{name}: {uri}")
        if not output and rendered_blocks:
            output = "\n".join(rendered_blocks)

    return output, exit_code


class CopilotEvent(ABC):
    """Base class for Copilot CLI JSONL events."""

    @abstractmethod
    def render(self, log_prefix: str) -> str | None:
        """Render the event as a string for terminal output."""

    @staticmethod
    def from_dict(data: dict[str, Any]) -> CopilotEvent | None:
        event_type = data.get("type")
        payload = _as_dict(data.get("data"))

        if event_type == "session.start":
            session_id_raw = payload.get("sessionId")
            session_id = session_id_raw if isinstance(session_id_raw, str) else None
            return SessionStartEvent(session_id=session_id)

        if event_type == "assistant.intent":
            intent = payload.get("intent")
            if isinstance(intent, str):
                return IntentEvent(intent=intent)
            return None

        if event_type == "assistant.message":
            message_id_raw = payload.get("messageId")
            message_id = message_id_raw if isinstance(message_id_raw, str) else None
            content_raw = payload.get("content")
            content = content_raw if isinstance(content_raw, str) else ""
            output_tokens_raw = payload.get("outputTokens")
            output_tokens = output_tokens_raw if isinstance(output_tokens_raw, int) else 0
            return MessageEvent(message_id=message_id, content=content, output_tokens=output_tokens)

        if event_type == "assistant.message_delta":
            message_id_raw = payload.get("messageId")
            message_id = message_id_raw if isinstance(message_id_raw, str) else None
            delta_raw = payload.get("deltaContent")
            delta = delta_raw if isinstance(delta_raw, str) else ""
            return MessageDeltaEvent(message_id=message_id, delta_content=delta)

        if event_type == "assistant.turn_end":
            turn_id_raw = payload.get("turnId")
            turn_id = turn_id_raw if isinstance(turn_id_raw, str) else None
            return TurnEndEvent(turn_id=turn_id)

        if event_type == "assistant.usage":
            model_raw = payload.get("model")
            model = model_raw if isinstance(model_raw, str) else ""
            return UsageEvent(
                model=model,
                input_tokens=int(payload.get("inputTokens") or 0),
                output_tokens=int(payload.get("outputTokens") or 0),
                cache_read_tokens=int(payload.get("cacheReadTokens") or 0),
                cache_write_tokens=int(payload.get("cacheWriteTokens") or 0),
                reasoning_tokens=int(payload.get("reasoningTokens") or 0),
            )

        if event_type == "tool.execution_start":
            tool_id_raw = payload.get("toolCallId")
            tool_id = tool_id_raw if isinstance(tool_id_raw, str) else None
            tool_name_raw = payload.get("toolName")
            tool_name = tool_name_raw if isinstance(tool_name_raw, str) else "Tool"
            return ToolUseEvent(
                tool_id=tool_id,
                tool_name=tool_name,
                arguments=payload.get("arguments"),
            )

        if event_type == "tool.execution_complete":
            tool_id_raw = payload.get("toolCallId")
            tool_id = tool_id_raw if isinstance(tool_id_raw, str) else None
            success = bool(payload.get("success"))
            output, exit_code = _extract_tool_result(payload.get("result"))
            error = _as_dict(payload.get("error"))
            error_message_raw = error.get("message")
            error_message = error_message_raw if isinstance(error_message_raw, str) else ""
            return ToolResultEvent(
                tool_id=tool_id,
                success=success,
                output=output,
                error_message=error_message,
                exit_code=exit_code,
            )

        if event_type == "session.error":
            error_type_raw = payload.get("errorType")
            error_type = error_type_raw if isinstance(error_type_raw, str) else None
            message_raw = payload.get("message")
            message = message_raw if isinstance(message_raw, str) else ""
            return ErrorEvent(message=message, error_type=error_type)

        if event_type == "result":
            session_id_raw = data.get("sessionId")
            session_id = session_id_raw if isinstance(session_id_raw, str) else None
            exit_code_raw = data.get("exitCode")
            exit_code = exit_code_raw if isinstance(exit_code_raw, int) else None
            usage = _as_dict(data.get("usage"))
            return ResultEvent(session_id=session_id, exit_code=exit_code, usage=usage)

        return None


class SessionStartEvent(CopilotEvent):
    """Session start event carrying the resumable session id."""

    def __init__(self, session_id: str | None):
        self.session_id = session_id

    def render(self, log_prefix: str) -> str | None:
        return None


class IntentEvent(CopilotEvent):
    """Ephemeral agent intent update."""

    def __init__(self, intent: str):
        self.intent = intent

    def render(self, log_prefix: str) -> str:
        return f"{log_prefix} \033[36m[Intent] {self.intent}\033[0m"


class MessageEvent(CopilotEvent):
    """Final assistant message content."""

    def __init__(self, message_id: str | None, content: str, output_tokens: int = 0):
        self.message_id = message_id
        self.content = content
        self.output_tokens = output_tokens

    def render(self, log_prefix: str) -> str | None:
        return self.content


class MessageDeltaEvent(CopilotEvent):
    """Streaming assistant message delta."""

    def __init__(self, message_id: str | None, delta_content: str):
        self.message_id = message_id
        self.delta_content = delta_content

    def render(self, log_prefix: str) -> str | None:
        return self.delta_content


class TurnEndEvent(CopilotEvent):
    """Turn completion marker."""

    def __init__(self, turn_id: str | None):
        self.turn_id = turn_id

    def render(self, log_prefix: str) -> str | None:
        return None


class UsageEvent(CopilotEvent):
    """Per-request usage summary."""

    def __init__(
        self,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        reasoning_tokens: int = 0,
    ):
        self.model = model
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_read_tokens = cache_read_tokens
        self.cache_write_tokens = cache_write_tokens
        self.reasoning_tokens = reasoning_tokens

    def render(self, log_prefix: str) -> str | None:
        return None


class ToolUseEvent(CopilotEvent):
    """Tool execution start event."""

    def __init__(self, tool_id: str | None, tool_name: str, arguments: Any):
        self.tool_id = tool_id
        self.tool_name = tool_name
        self.arguments = arguments

    def render(self, log_prefix: str) -> str:
        truncated = truncate_tool_params(self.tool_name, self.arguments)
        return f"{log_prefix} \033[34m[Tool Use] {self.tool_name} {truncated}\033[0m"


class ToolResultEvent(CopilotEvent):
    """Tool execution completion event."""

    def __init__(
        self,
        tool_id: str | None,
        success: bool,
        output: str = "",
        error_message: str = "",
        exit_code: int | None = None,
    ):
        self.tool_id = tool_id
        self.success = success
        self.output = output
        self.error_message = error_message
        self.exit_code = exit_code
        self.tool_name_resolved: str = "Tool"

    def render(self, log_prefix: str) -> str:
        if self.output:
            truncated = truncate_content(self.output)
            return f"{log_prefix} \033[32m[Tool Result] {truncated}\033[0m"
        if self.error_message:
            return f"{log_prefix} \033[31m[Tool Error] {self.error_message}\033[0m"
        if self.success:
            return f"{log_prefix} \033[32m{self.tool_name_resolved} ran successfully\033[0m"
        return f"{log_prefix} \033[31m[Tool Error] {self.tool_name_resolved} failed\033[0m"


class ErrorEvent(CopilotEvent):
    """Session-level error event."""

    def __init__(self, message: str, error_type: str | None = None):
        self.message = message
        self.error_type = error_type

    def render(self, log_prefix: str) -> str:
        label = f"{self.error_type}: " if self.error_type else ""
        return f"{log_prefix} \033[31m[Error] {label}{self.message}\033[0m"


class ResultEvent(CopilotEvent):
    """Top-level non-interactive result summary."""

    def __init__(self, session_id: str | None, exit_code: int | None, usage: dict[str, Any] | None = None):
        self.session_id = session_id
        self.exit_code = exit_code
        self.usage = usage or {}

    def render(self, log_prefix: str) -> str | None:
        return None
