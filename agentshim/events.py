from typing import Any, Protocol


class AgentEventHandler(Protocol):
    """Protocol for handling agent events."""

    def on_thinking(self, text: str) -> None:
        """Handle agent thinking output."""
        ...

    def on_tool_call(self, tool: str, args: dict[str, Any] | str | None = None) -> None:
        """Handle tool execution start."""
        ...

    def on_tool_result(
        self,
        tool: str,
        stdout: str = "",
        stderr: str = "",
        exit_code: int | None = None,
        duration: float | None = None,
    ) -> None:
        """Handle tool execution result."""
        ...

    def on_usage(self, usage: dict[str, Any]) -> None:
        """Handle a per-turn usage update from the underlying CLI."""
        ...
