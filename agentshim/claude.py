import json
import subprocess
import time
from collections.abc import Callable
from typing import Any

from .trajectory import TrajectoryRecorderProtocol

from .base import register_provider
from .claude_events import (
    ClaudeEvent,
    MultiEvent,
    ResultEvent,
    TextEvent,
    ToolResultEvent,
    ToolUseEvent,
)
from .cli_agent import CLICodingAgent, CLIGenerationSession
from .events import AgentEventHandler
from .mcp_config import HttpMcpServer, McpServerConfig
from .sandbox import SandboxConfig, build_claude_sandbox_settings, resolve_sandbox


class ClaudeGenerationSession(CLIGenerationSession):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        # Initialize state required for stream processing
        self.tool_map: dict[str, str] = {}
        self.tool_start_times: dict[str, float] = {}
        self.tool_args: dict[str, Any] = {}
        self.final_result: str | None = None

    def _process_stdout(self, line: str) -> None:
        """Process a line from stdout."""
        if not line:
            return
        try:
            data = json.loads(line)
            event = ClaudeEvent.from_dict(data)
            if event:
                self._handle_event(event)
        except json.JSONDecodeError:
            # Fallback for non-JSON lines - still accumulate them
            self.stdout_lines.append(line.rstrip())
            if not self.silent:
                if self._at_line_start:
                    self._log_raw(f"{self.log_prefix} ")
                self._log_raw(line.rstrip() + "\n")
                self._at_line_start = True

    def _handle_event(self, event: ClaudeEvent):
        """Handle a single parsed Claude event."""
        # Handle MultiEvent by processing each sub-event
        if isinstance(event, MultiEvent):
            for sub_event in event.events:
                self._handle_event(sub_event)
            return

        # 1. Update State (Accumulator, Tool Map, Context Injection)
        self._update_state(event)

        # 2. Render Output
        if not self.silent:
            self._render_event(event)

    def _update_state(self, event: ClaudeEvent):
        """Update internal state based on the event."""
        if isinstance(event, TextEvent):
            self.stdout_lines.append(event.text)
            if self.event_handler:
                self.event_handler.on_thinking(event.text)

        elif isinstance(event, ToolUseEvent):
            if event.tool_id:
                self.tool_map[event.tool_id] = event.tool_name
                self.tool_start_times[event.tool_id] = time.time()
                self.tool_args[event.tool_id] = event.parameters
                if self.event_handler:
                    self.event_handler.on_tool_call(event.tool_name, event.parameters)

        elif isinstance(event, ToolResultEvent):
            if event.tool_id:
                # Inject resolved name into the event for rendering
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

        elif isinstance(event, ResultEvent):
            # Store the final result from the result event
            self.final_result = event.result

    def _render_event(self, event: ClaudeEvent):
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

    def run(self, prompt: str) -> str:
        """Execute the command and return the result."""
        super().run(prompt)
        # Return the final result if available, otherwise join accumulated text
        if self.final_result:
            return self.final_result
        return "\n".join(self.stdout_lines)


@register_provider("claude", "claude-code", "anthropic")
class ClaudeCodeCodingAgent(CLICodingAgent):
    """Coding agent implementation using the Claude Code CLI tool."""

    def __init__(
        self,
        model: str | None = None,
        recorder: TrajectoryRecorderProtocol | None = None,
        event_handler: AgentEventHandler | None = None,
        mcp_servers: list[McpServerConfig] | None = None,
        sandbox: bool | SandboxConfig = False,
    ):
        """Initialize the Claude Code coding agent.

        Args:
            model: Optional model name to use with Claude Code. If None, uses default.
            recorder: Trajectory recorder instance.
            event_handler: Optional event handler for UI updates.
            mcp_servers: Optional list of MCP server configurations.
            sandbox: If True (or a ``SandboxConfig``), enable Claude Code's
                native sandbox (bubblewrap on Linux / Seatbelt on macOS)
                by injecting a ``sandbox`` settings block via
                ``--settings``. Only bash subprocess commands are
                sandboxed; the Claude process itself is not wrapped.
                Defaults to False (no sandbox).
        """
        super().__init__("claude", model, recorder, event_handler, mcp_servers)
        self.sandbox = resolve_sandbox(sandbox)
        if self.sandbox is not None:
            # Without this, Claude Code cd's into a per-invocation scratch dir
            # under <project-root>/.local_tmp/claude-$UID/cwd-* before every
            # Bash call. That dir is outside the sandbox's allow_write set, so
            # every sandboxed Bash invocation fails with EROFS before its
            # command runs. Maintaining the project cwd avoids the scratch dir.
            self.env["CLAUDE_BASH_MAINTAIN_PROJECT_WORKING_DIR"] = "1"

    @property
    def claude_path(self) -> str:
        """Return path to claude binary (for backward compatibility)."""
        return self.binary_path

    @property
    def _log_prefix(self) -> str:
        """Return the log prefix for this agent."""
        return "[Claude]"

    def _build_mcp_config_json(self) -> str:
        """Build the JSON string for --mcp-config."""
        servers: dict[str, dict[str, Any]] = {}
        for s in self.mcp_servers:
            if isinstance(s, HttpMcpServer):
                servers[s.name] = {"url": s.url}
            else:
                entry: dict[str, Any] = {"command": s.command, "args": s.args}
                if s.env:
                    entry["env"] = s.env
                servers[s.name] = entry
        return json.dumps({"mcpServers": servers})

    def _get_command(self, prompt: str) -> list[str]:
        cmd = [
            self.binary_path,
            "-p",  # Print mode, accepts prompt from stdin
            "--dangerously-skip-permissions",  # Auto-approval mode
            "--output-format",
            "stream-json",
            "--verbose",
            prompt,
        ]
        if self.model:
            cmd.extend(["--model", self.model])
        if self.mcp_servers:
            cmd.extend(
                [
                    "--mcp-config",
                    self._build_mcp_config_json(),
                    "--strict-mcp-config",
                ]
            )
        if self.sandbox is not None:
            cmd.extend(["--settings", json.dumps(build_claude_sandbox_settings(self.sandbox))])
        return cmd

    def _create_session(
        self,
        cmd: list[str],
        cwd: str | None = None,
        timeout: int = 300,
        silent: bool = False,
        recorder: TrajectoryRecorderProtocol | None = None,
        on_process_started: Callable[[subprocess.Popen[str]], None] | None = None,
    ) -> ClaudeGenerationSession:
        return ClaudeGenerationSession(
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
