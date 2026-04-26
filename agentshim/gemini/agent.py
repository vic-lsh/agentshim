import json
import time
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from ..base import register_provider
from ..cli_agent import CLICodingAgent, CLIGenerationSession
from ..events import AgentEventHandler
from ..executor import CommandExecutor, CommandHandle
from ..sandbox import SandboxConfig
from ..usage import ProviderUsage, TokenUsage
from .events import GeminiEvent, InitEvent, MessageEvent, ToolResultEvent, ToolUseEvent


class GeminiGenerationSession(CLIGenerationSession):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        # Initialize state required for stream processing
        self.tool_map: dict[str, str] = {}
        self.tool_start_times: dict[str, float] = {}
        self.tool_args: dict[str, Any] = {}
        # Gemini's stream-json does not emit token usage. We count
        # assistant MessageEvents as a turn proxy; token fields stay 0.
        self._assistant_message_count: int = 0

    def _process_stdout(self, line: str) -> None:
        """Process a line from stdout."""
        if not line:
            return
        try:
            data = json.loads(line)
            event = GeminiEvent.from_dict(data)
            if event:
                self._handle_event(event)
        except json.JSONDecodeError:
            stripped = line.rstrip()
            if stripped:
                self.event_handler.on_thinking(stripped + "\n")

    def _handle_event(self, event: GeminiEvent):
        """Handle a single parsed Gemini event."""
        self._update_state(event)

    def _update_state(self, event: GeminiEvent):
        """Update internal state based on the event."""
        if isinstance(event, InitEvent):
            if self.session_id is None and event.session_id:
                self.session_id = event.session_id
            return

        if isinstance(event, MessageEvent):
            if event.role == "assistant":
                self.stdout_lines.append(event.content)
                self._assistant_message_count += 1
                self.usage = ProviderUsage(
                    tokens=TokenUsage(turns=self._assistant_message_count),
                    provider="gemini",
                )
                if self.event_handler:
                    self.event_handler.on_thinking(event.content)

        elif isinstance(event, ToolUseEvent):
            if event.tool_id:
                self.tool_map[event.tool_id] = event.tool_name
                self.tool_start_times[event.tool_id] = time.time()
                self.tool_args[event.tool_id] = event.parameters
            if self.event_handler:
                self.event_handler.on_tool_call(event.tool_name, event.parameters)

        elif isinstance(event, ToolResultEvent) and event.tool_id:
            event.tool_name_resolved = self.tool_map.get(event.tool_id, "Tool")

            start_time = self.tool_start_times.get(event.tool_id)
            duration = time.time() - start_time if start_time else None

            if self.event_handler:
                self.event_handler.on_tool_result(
                    tool=event.tool_name_resolved,
                    stdout=event.output,
                    duration=duration,
                )


@register_provider("gemini")
class GeminiCodingAgent(CLICodingAgent):
    """Coding agent implementation using the Gemini CLI tool."""

    def __init__(
        self,
        model: str | None = None,
        event_handler: AgentEventHandler | None = None,
        event_handlers: Iterable[AgentEventHandler] | None = None,
        mcp_servers: Sequence[object] | None = None,
        sandbox: bool | SandboxConfig = False,
        executor: CommandExecutor | None = None,
    ):
        """Initialize the Gemini coding agent.

        Args:
            model: Optional model name to use.
            event_handler: Optional event handler for UI updates.
            mcp_servers: Optional list of MCP server configurations.
            sandbox: Not supported for Gemini; must be False.
            executor: Optional command executor for binary lookup and process execution.

        Raises:
            ValueError: If mcp_servers is non-empty (not supported).
            NotImplementedError: If ``sandbox`` is truthy.
        """
        if mcp_servers:
            raise ValueError("GeminiCodingAgent does not support programmatic MCP server configuration via CLI flags")
        if sandbox:
            raise NotImplementedError("sandbox is not supported for GeminiCodingAgent")
        super().__init__("gemini", model, event_handler, event_handlers, executor=executor)

    @property
    def gemini_path(self) -> str:
        """Return path to gemini binary (for backward compatibility)."""
        return self.binary_path

    @property
    def _log_prefix(self) -> str:
        """Return the log prefix for this agent."""
        return "[Gemini]"

    def _get_command(self, prompt: str, resume_session_id: str | None = None) -> list[str]:
        cmd = [self.binary_path]

        cmd.extend(["-y"])

        if self.model:
            cmd.extend(["--model", self.model])

        cmd.extend(["-o", "stream-json"])

        if resume_session_id:
            cmd.extend(["--resume", resume_session_id])

        return cmd

    def _create_session(
        self,
        cmd: list[str],
        cwd: str | None = None,
        timeout: int = 300,
        silent: bool = False,
        on_process_started: Callable[[CommandHandle], None] | None = None,
    ) -> GeminiGenerationSession:
        return GeminiGenerationSession(
            binary_name=self.binary_name,
            env=self.env,
            log_prefix=self._log_prefix,
            cmd=cmd,
            logger=self.logger,
            cwd=cwd,
            timeout=timeout,
            silent=silent,
            event_handler=self.event_handler,
            executor=self.executor,
            on_process_started=on_process_started,
        )
