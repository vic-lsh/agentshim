import json
import subprocess
import time
from collections.abc import Callable, Iterable
from typing import Any

from ..base import register_provider
from ..cli_agent import CLICodingAgent, CLIGenerationSession
from ..events import AgentEventHandler
from ..mcp_config import HttpMcpServer, McpServerConfig
from ..sandbox import SandboxConfig, build_claude_sandbox_settings, resolve_sandbox
from ..usage import ProviderUsage, TokenUsage
from .events import (
    ClaudeEvent,
    MultiEvent,
    ResultEvent,
    SystemEvent,
    TextEvent,
    ToolResultEvent,
    ToolUseEvent,
)


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
            if line.rstrip():
                self.event_handler.on_thinking(line.rstrip() + "\n")

    def _handle_event(self, event: ClaudeEvent):
        """Handle a single parsed Claude event."""
        if isinstance(event, MultiEvent):
            if event.usage and self.event_handler is not None:
                on_usage = getattr(self.event_handler, "on_usage", None)
                if on_usage is not None:
                    on_usage(event.usage)
            for sub_event in event.events:
                self._handle_event(sub_event)
            return

        self._update_state(event)

    def _update_state(self, event: ClaudeEvent):
        """Update internal state based on the event."""
        if isinstance(event, SystemEvent):
            if self.session_id is None and event.session_id:
                self.session_id = event.session_id
            return

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
                event.tool_name_resolved = self.tool_map.get(event.tool_id, "Tool")

                start_time = self.tool_start_times.get(event.tool_id)
                duration = time.time() - start_time if start_time else None

                if self.event_handler:
                    self.event_handler.on_tool_result(
                        tool=event.tool_name_resolved,
                        stdout=event.output,
                        duration=duration,
                    )

        elif isinstance(event, ResultEvent):
            self.final_result = event.result
            self.final_usage = event.usage
            self.total_cost_usd = event.total_cost_usd
            self.duration_ms = event.duration_ms
            # Anthropic reports cache_creation + cache_read as disjoint
            # from input_tokens; fold them into input_tokens to match the
            # crucible invariant (cached ⊆ input).
            usage = event.usage or {}
            cached = int(usage.get("cache_creation_input_tokens") or 0) + int(usage.get("cache_read_input_tokens") or 0)
            self.usage = ProviderUsage(
                tokens=TokenUsage(
                    input_tokens=int(usage.get("input_tokens") or 0) + cached,
                    output_tokens=int(usage.get("output_tokens") or 0),
                    cached_input_tokens=cached,
                    turns=int(event.num_turns or 0),
                ),
                total_cost_usd=event.total_cost_usd,
                provider="claude",
            )

    def run(self, prompt: str) -> str:
        """Execute the command and return the result."""
        super().run(prompt)
        if self.final_result:
            return self.final_result
        return "\n".join(self.stdout_lines)


@register_provider("claude", "claude-code", "anthropic")
class ClaudeCodeCodingAgent(CLICodingAgent):
    """Coding agent implementation using the Claude Code CLI tool."""

    def __init__(
        self,
        model: str | None = None,
        event_handler: AgentEventHandler | None = None,
        event_handlers: Iterable[AgentEventHandler] | None = None,
        mcp_servers: list[McpServerConfig] | None = None,
        sandbox: bool | SandboxConfig = False,
    ):
        """Initialize the Claude Code coding agent.

        Args:
            model: Optional model name to use with Claude Code. If None, uses default.
            event_handler: Optional event handler for UI updates.
            mcp_servers: Optional list of MCP server configurations.
            sandbox: If True (or a ``SandboxConfig``), enable Claude Code's
                native sandbox (bubblewrap on Linux / Seatbelt on macOS)
                by injecting a ``sandbox`` settings block via
                ``--settings``. Only bash subprocess commands are
                sandboxed; the Claude process itself is not wrapped.
                Defaults to False (no sandbox).
        """
        super().__init__("claude", model, event_handler, event_handlers, mcp_servers)
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
        """Build the JSON string for --mcp-config.

        Claude Code runs the rendered config through ``--strict-mcp-config``
        validation, which requires HTTP servers to declare ``type``
        explicitly (``"sse"`` or ``"http"``). ``HttpMcpServer`` represents
        the SSE transport (the field doc says HTTP/SSE; current call sites
        use ``…/sse`` URLs), so emit ``type: "sse"``. ``headers`` is
        included only when non-empty, mirroring the schema's optional
        nature.
        """
        servers: dict[str, dict[str, Any]] = {}
        for server in self.mcp_servers:
            if isinstance(server, HttpMcpServer):
                http_entry: dict[str, Any] = {"type": "sse", "url": server.url}
                if server.headers:
                    http_entry["headers"] = dict(server.headers)
                servers[server.name] = http_entry
            else:
                entry: dict[str, Any] = {"command": server.command, "args": server.args}
                if server.env:
                    entry["env"] = server.env
                servers[server.name] = entry
        return json.dumps({"mcpServers": servers})

    def _get_command(self, prompt: str, resume_session_id: str | None = None) -> list[str]:
        cmd = [
            self.binary_path,
            "-p",  # Print mode, accepts prompt from stdin
            "--dangerously-skip-permissions",  # Auto-approval mode
            "--output-format",
            "stream-json",
            "--verbose",
        ]
        if resume_session_id:
            cmd.extend(["--resume", resume_session_id])
        cmd.append(prompt)
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
            event_handler=self.event_handler,
            on_process_started=on_process_started,
        )
