import json
import time
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from ..base import register_provider
from ..cli_agent import CLICodingAgent, CLIGenerationSession
from ..events import AgentEventHandler
from ..executor import CommandExecutor, CommandHandle
from ..mcp_config import HttpMcpServer, McpServerConfig
from ..sandbox import SandboxConfig
from ..usage import ProviderUsage, TokenUsage
from .events import (
    CodexEvent,
    ErrorEvent,
    LifecycleEvent,
    TextEvent,
    ThreadStartedEvent,
    ToolResultEvent,
    ToolUseEvent,
    TurnCompletedEvent,
)


class CodexGenerationSession(CLIGenerationSession):
    """Session that parses Codex ``--json`` event stream."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.tool_map: dict[str, str] = {}
        self.tool_start_times: dict[str, float] = {}
        self.tool_args: dict[str, Any] = {}
        # Codex has no single "final message" frame; track the most recent
        # agent_message text so run() can return it as the final result.
        self.final_result: str | None = None
        # Accumulator for per-turn usage; finalized into self.usage at end.
        self._accumulated_tokens = TokenUsage()

    def _process_stdout(self, line: str) -> None:
        if not line.strip():
            return
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            stripped = line.rstrip()
            self.stdout_lines.append(stripped)
            if stripped:
                self.event_handler.on_thinking(stripped)
            return

        event = CodexEvent.from_dict(data)
        if event is None:
            stripped = line.rstrip()
            if self.event_handler and stripped:
                self.event_handler.on_thinking(stripped)
            self.stdout_lines.append(stripped)
            return
        self._handle_event(event)

    def _handle_event(self, event: CodexEvent):
        self._update_state(event)

    def _update_state(self, event: CodexEvent):
        if isinstance(event, ThreadStartedEvent):
            if self.session_id is None and event.thread_id:
                self.session_id = event.thread_id
            if self.event_handler and event.thread_id:
                self.event_handler.on_thinking(f"[codex thread {event.thread_id} started]")
            return

        if isinstance(event, LifecycleEvent):
            if self.event_handler and event.event_type == "turn.started":
                self.event_handler.on_thinking("[codex turn started]")
            return

        if isinstance(event, TextEvent):
            if event.text:
                self.stdout_lines.append(event.text)
                self.final_result = event.text
                if self.event_handler:
                    self.event_handler.on_thinking(event.text)
            return

        if isinstance(event, ToolUseEvent):
            if event.tool_name == "execute" and event.tool_id:
                self.tool_map[event.tool_id] = event.tool_name
                self.tool_start_times[event.tool_id] = time.time()
                self.tool_args[event.tool_id] = event.parameters
            if self.event_handler:
                self.event_handler.on_tool_call(event.tool_name, event.parameters)
            return

        if isinstance(event, TurnCompletedEvent):
            self._accumulated_tokens = self._accumulated_tokens + TokenUsage(
                input_tokens=event.input_tokens,
                output_tokens=event.output_tokens,
                cached_input_tokens=event.cached_input_tokens,
                turns=1,
            )
            self.usage = ProviderUsage(
                tokens=self._accumulated_tokens,
                total_cost_usd=None,
                provider="codex",
            )
            normalized_usage = {
                "input_tokens": event.input_tokens,
                "output_tokens": event.output_tokens,
                "cache_read_input_tokens": event.cached_input_tokens,
                "cache_creation_input_tokens": 0,
            }
            if event.has_usage:
                self.final_usage = normalized_usage
            if self.event_handler is not None and event.has_usage:
                on_usage = getattr(self.event_handler, "on_usage", None)
                if on_usage is not None:
                    on_usage(normalized_usage)
                self.event_handler.on_thinking(
                    f"[codex turn complete: in={normalized_usage['input_tokens']} "
                    f"cached={normalized_usage['cache_read_input_tokens']} "
                    f"out={normalized_usage['output_tokens']}]"
                )
            elif self.event_handler is not None:
                self.event_handler.on_thinking("[codex turn complete]")
            return

        if isinstance(event, ToolResultEvent):
            if not event.tool_id:
                if event.tool_name is None:
                    return
                event.tool_name_resolved = event.tool_name
                args = event.parameters or {}
                duration = None
                if self.event_handler:
                    self.event_handler.on_tool_call(event.tool_name, args)
            else:
                event.tool_name_resolved = self.tool_map.get(
                    event.tool_id,
                    event.tool_name or "Tool",
                )

                start_time = self.tool_start_times.get(event.tool_id)
                duration = time.time() - start_time if start_time else None
                args = self.tool_args.get(event.tool_id, event.parameters or {})

                if event.tool_id not in self.tool_map and self.event_handler and event.tool_name:
                    self.event_handler.on_tool_call(event.tool_name, args)

            if self.event_handler:
                self.event_handler.on_tool_result(
                    tool=event.tool_name_resolved,
                    stdout=event.output,
                    exit_code=event.exit_code,
                    duration=duration,
                )
            return

        if isinstance(event, ErrorEvent):
            self.stdout_lines.append(event.message)
            if event.message:
                self.event_handler.on_thinking(f"[codex error] {event.message}")

    def _process_stderr(self, line: str) -> None:
        line_stripped = line.rstrip("\n")
        self.stderr_lines.append(line)
        if line_stripped:
            self.event_handler.on_thinking(f"[codex stderr] {line_stripped}")

    def run(self, prompt: str) -> str:
        started = time.monotonic()
        try:
            super().run(prompt)
        finally:
            self.duration_ms = int((time.monotonic() - started) * 1000)
        if self.final_result:
            return self.final_result
        return "\n".join(self.stdout_lines)


@register_provider("codex", aliases=("openai",))
class CodexCodingAgent(CLICodingAgent):
    """Coding agent implementation using the Codex CLI tool."""

    def __init__(
        self,
        model: str | None = None,
        event_handler: AgentEventHandler | None = None,
        event_handlers: Iterable[AgentEventHandler] | None = None,
        mcp_servers: Sequence[McpServerConfig] | None = None,
        sandbox: bool | SandboxConfig = False,
        executor: CommandExecutor | None = None,
    ):
        """Initialize the Codex coding agent.

        Args:
            model: Optional model name to use with codex. If None, uses default.
            event_handler: Optional event handler for UI updates.
            mcp_servers: Optional list of MCP server configurations.
            sandbox: Not supported for Codex; must be False.
            executor: Optional command executor for binary lookup and process execution.
        """
        if sandbox:
            raise NotImplementedError("sandbox is not supported for CodexCodingAgent")
        super().__init__("codex", model, event_handler, event_handlers, mcp_servers, executor=executor)

    @property
    def codex_path(self) -> str:
        """Return path to codex binary (for backward compatibility)."""
        return self.binary_path

    @property
    def _log_prefix(self) -> str:
        """Return the log prefix for this agent."""
        return "[Codex]"

    def _build_mcp_args(self) -> list[str]:
        """Build -c flag arguments for MCP server configuration."""
        args: list[str] = []
        for server in self.mcp_servers:
            prefix = f"mcp_servers.{server.name}"
            if isinstance(server, HttpMcpServer):
                args.extend(["-c", f'{prefix}.url="{server.url}"'])
            else:
                args.extend(["-c", f'{prefix}.command="{server.command}"'])
                if server.args:
                    toml_arr = "[" + ", ".join(f'"{arg}"' for arg in server.args) + "]"
                    args.extend(["-c", f"{prefix}.args={toml_arr}"])
                for key, value in server.env.items():
                    args.extend(["-c", f'{prefix}.env.{key}="{value}"'])
        return args

    def _get_command(self, prompt: str, resume_session_id: str | None = None) -> list[str]:
        cmd: list[str] = [self.binary_path, "exec"]
        if resume_session_id:
            cmd.extend(["resume", resume_session_id])
        cmd.extend(["--dangerously-bypass-approvals-and-sandbox", "--json"])
        if self.model:
            cmd.extend(["--model", self.model])
        if self.mcp_servers:
            cmd.extend(self._build_mcp_args())
        # Tell Codex to read the prompt from stdin instead of expecting an
        # inline positional prompt argument.
        cmd.append("-")
        return cmd

    def _create_session(
        self,
        cmd: list[str],
        cwd: str | None = None,
        timeout: int = 300,
        silent: bool = False,
        on_process_started: Callable[[CommandHandle], None] | None = None,
    ) -> CodexGenerationSession:
        return CodexGenerationSession(
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
