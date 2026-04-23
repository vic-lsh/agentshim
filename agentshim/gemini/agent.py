import json
import logging
import subprocess
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import agentshim.trajectory as _trajectory_module
from agentshim.trajectory import TrajectoryRecorderProtocol

from ..base import register_provider
from ..cli_agent import CLICodingAgent, CLIGenerationSession
from ..events import AgentEventHandler
from ..sandbox import SandboxConfig
from ..usage import ProviderUsage, TokenUsage
from .events import GeminiEvent, InitEvent, MessageEvent, ToolResultEvent, ToolUseEvent

_logger = logging.getLogger(__name__)


class GeminiGenerationSession(CLIGenerationSession):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        # Initialize state required for stream processing
        self.tool_map: dict[str, str] = {}
        self.tool_start_times: dict[str, float] = {}
        self.tool_args: dict[str, Any] = {}
        # Capture call_id and run_id for correlation
        self.call_id = _trajectory_module.get_current_call_id()
        self.run_id = _trajectory_module.get_run_id()
        # Gemini's stream-json does not emit token usage. We count
        # assistant MessageEvents as a turn proxy; token fields stay 0.
        self._assistant_message_count: int = 0

    def _write_call_metadata(self):
        """Write metadata file to help correlate Gemini session with trajectory call."""
        if self.call_id is None or self.run_id is None:
            return

        try:
            gemini_tmp_dir = Path.home() / ".gemini" / "tmp"
            if not gemini_tmp_dir.exists():
                return

            if self.cwd:
                project_name = Path(self.cwd).name
                project_dir = gemini_tmp_dir / project_name / "chats"
                if project_dir.exists():
                    metadata_file = project_dir / f"sds_call_{self.call_id:03d}.json"
                    metadata = {
                        "call_id": self.call_id,
                        "run_id": self.run_id,
                        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "cwd": self.cwd,
                    }
                    with open(metadata_file, "w") as f:
                        json.dump(metadata, f, indent=2)
        except Exception:
            _logger.debug("Failed to write Gemini call metadata", exc_info=True)

    def run(self, prompt: str) -> str:
        """Execute the generation process, writing call metadata first."""
        self._write_call_metadata()

        if not self.silent:
            self._log_raw(f"{self.log_prefix} Input Prompt:\n{prompt}\n")

        return super().run(prompt)

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
            if not self.silent:
                if self._at_line_start:
                    self._log_raw(f"{self.log_prefix} ")
                self._log_raw(line.rstrip() + "\n")
                self._at_line_start = True

    def _handle_event(self, event: GeminiEvent):
        """Handle a single parsed Gemini event."""
        self._update_state(event)

        if not self.silent:
            self._render_event(event)

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
            args = self.tool_args.get(event.tool_id, {})

            self.recorder.add_tool_call(
                tool=event.tool_name_resolved,
                args=args,
                stdout=event.output,
                duration=duration,
            )

            if self.event_handler:
                self.event_handler.on_tool_result(
                    tool=event.tool_name_resolved,
                    stdout=event.output,
                    duration=duration,
                )

    def _render_event(self, event: GeminiEvent):
        """Render the event to stdout."""
        if isinstance(event, MessageEvent):
            if event.role == "assistant":
                self._print_stream_content(event.content)
            return

        if not self._at_line_start:
            self._log_raw("\n")
            self._at_line_start = True

        output = event.render(self.log_prefix)
        if output:
            self._log_raw(output + "\n")


@register_provider("gemini")
class GeminiCodingAgent(CLICodingAgent):
    """Coding agent implementation using the Gemini CLI tool."""

    def __init__(
        self,
        model: str | None = None,
        recorder: TrajectoryRecorderProtocol | None = None,
        event_handler: AgentEventHandler | None = None,
        mcp_servers: Sequence[object] | None = None,
        sandbox: bool | SandboxConfig = False,
    ):
        """Initialize the Gemini coding agent.

        Args:
            model: Optional model name to use.
            recorder: Trajectory recorder instance.
            event_handler: Optional event handler for UI updates.
            mcp_servers: Optional list of MCP server configurations.
            sandbox: Not supported for Gemini; must be False.

        Raises:
            ValueError: If mcp_servers is non-empty (not supported).
            NotImplementedError: If ``sandbox`` is truthy.
        """
        if mcp_servers:
            raise ValueError("GeminiCodingAgent does not support programmatic MCP server configuration via CLI flags")
        if sandbox:
            raise NotImplementedError("sandbox is not supported for GeminiCodingAgent")
        super().__init__("gemini", model, recorder, event_handler)

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
        recorder: TrajectoryRecorderProtocol | None = None,
        on_process_started: Callable[[subprocess.Popen[str]], None] | None = None,
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
            recorder=recorder,
            event_handler=self.event_handler,
            on_process_started=on_process_started,
        )
