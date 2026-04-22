import json
import subprocess
from collections.abc import Callable
from typing import Any

from .trajectory import TrajectoryRecorderProtocol

from .base import register_provider
from .cli_agent import CLICodingAgent, CLIGenerationSession
from .events import AgentEventHandler
from .opencode_events import OpencodeEvent, TextEvent, ToolUseEvent
from .sandbox import SandboxConfig

OPENCODE_DEFAULT_MODEL = "google-vertex/gemini-3-pro-preview"


def _to_args_dict(input_data: Any) -> dict[str, Any]:
    """Convert input_data to a dict[str, Any] for tool call recording."""
    if isinstance(input_data, dict):
        return {str(k): v for k, v in input_data.items()}  # type: ignore[reportUnknownVariableType]
    return {"input": input_data}


class OpencodeGenerationSession(CLIGenerationSession):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def _process_stdout(self, line: str) -> None:
        """Process a line from stdout."""
        if not line:
            return
        try:
            data = json.loads(line)
            event = OpencodeEvent.from_dict(data)
            if event:
                self._handle_event(event)
        except json.JSONDecodeError:
            # Fallback for non-JSON lines
            if not self.silent:
                if self._at_line_start:
                    self._log_raw(f"{self.log_prefix} ")
                self._log_raw(line.rstrip() + "\n")
                self._at_line_start = True

    def _handle_event(self, event: OpencodeEvent):
        """Handle a single parsed Opencode event."""
        # 1. Update State
        if isinstance(event, TextEvent):
            self.stdout_lines.append(event.text)
            if self.event_handler:
                self.event_handler.on_thinking(event.text)

        elif isinstance(event, ToolUseEvent):
            # Record tool call if it has a completion status
            if event.status in ("success", "error"):
                args = _to_args_dict(event.input_data)
                stdout = str(event.output_data) if event.output_data is not None else ""

                self.recorder.add_tool_call(
                    tool=event.tool_name,
                    args=args,
                    stdout=stdout,
                    # duration is not easily available from event stream
                )

                if self.event_handler:
                    # Emit both call and result since we only capture completion
                    self.event_handler.on_tool_call(event.tool_name, args)
                    self.event_handler.on_tool_result(
                        tool=event.tool_name,
                        stdout=stdout,
                        # No duration available
                    )

        # 2. Render Output
        if not self.silent:
            self._render_event(event)

    def _render_event(self, event: OpencodeEvent):
        """Render the event to stdout."""
        # Handle streaming text differently from block events
        if isinstance(event, TextEvent):
            self._print_stream_content(event.text)
            return

        # Ensure we start block events on a new line
        if not self._at_line_start:
            self._log_raw("\n")
            self._at_line_start = True

        # Render and print
        output = event.render(self.log_prefix)
        if output:
            self._log_raw(output + "\n")


@register_provider("opencode")
class OpencodeCodingAgent(CLICodingAgent):
    """Coding agent implementation using the Opencode CLI tool."""

    def __init__(
        self,
        model: str | None = None,
        recorder: TrajectoryRecorderProtocol | None = None,
        event_handler: AgentEventHandler | None = None,
        mcp_servers: list[object] | None = None,
        sandbox: bool | SandboxConfig = False,
    ):
        """Initialize the Opencode coding agent.

        Args:
            model: Optional model name to use.
            recorder: Trajectory recorder instance.
            event_handler: Optional event handler for UI updates.
            mcp_servers: Optional list of MCP server configurations.
            sandbox: Not supported for Opencode; must be False.

        Raises:
            ValueError: If mcp_servers is non-empty (not supported).
            NotImplementedError: If ``sandbox`` is truthy.
        """
        if mcp_servers:
            raise ValueError("OpencodeCodingAgent does not support programmatic MCP server configuration via CLI flags")
        if sandbox:
            raise NotImplementedError("sandbox is not supported for OpencodeCodingAgent")
        if not model:
            model = OPENCODE_DEFAULT_MODEL
        super().__init__("opencode", model, recorder, event_handler)

    @property
    def _log_prefix(self) -> str:
        """Return the log prefix for this agent."""
        return "[Opencode]"

    def _get_command(self, prompt: str) -> list[str]:
        cmd = [self.binary_path, "run", f'"{prompt}"']

        if self.model:
            cmd.extend(["--model", self.model])

        # Output in json format
        cmd.extend(["--format=json"])

        return cmd

    def _create_session(
        self,
        cmd: list[str],
        cwd: str | None = None,
        timeout: int = 300,
        silent: bool = False,
        recorder: TrajectoryRecorderProtocol | None = None,
        on_process_started: Callable[[subprocess.Popen[str]], None] | None = None,
    ) -> OpencodeGenerationSession:
        return OpencodeGenerationSession(
            binary_name=self.binary_name,
            env=self.env,
            log_prefix=self._log_prefix,
            cmd=cmd,
            logger=self.logger,
            cwd=cwd,
            timeout=timeout,
            silent=silent,
            recorder=recorder,
            event_handler=self.event_handler,
            on_process_started=on_process_started,
        )
