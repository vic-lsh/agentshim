import json
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from ..base import register_provider
from ..cli_agent import CLICodingAgent, CLIGenerationSession
from ..events import AgentEventHandler
from ..executor import CommandExecutor, CommandHandle
from ..sandbox import SandboxConfig
from ..usage import ProviderUsage, TokenUsage
from .events import OpencodeEvent, StepFinishEvent, TextEvent, ToolUseEvent

OPENCODE_DEFAULT_MODEL = "google-vertex/gemini-3-pro-preview"


def _to_args_dict(input_data: Any) -> dict[str, Any]:
    """Convert input_data to a dict[str, Any] for tool call recording."""
    if isinstance(input_data, dict):
        return {str(k): v for k, v in input_data.items()}  # type: ignore[reportUnknownVariableType]
    return {"input": input_data}


class OpencodeGenerationSession(CLIGenerationSession):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        # Accumulates usage across step_finish events.
        self._accumulated_tokens = TokenUsage()
        self._accumulated_cost_usd: float = 0.0
        self._saw_cost: bool = False

    def _process_stdout(self, line: str) -> None:
        """Process a line from stdout."""
        if not line:
            return
        try:
            data = json.loads(line)
            if self.session_id is None:
                sid = data.get("sessionID")
                if isinstance(sid, str) and sid:
                    self.session_id = sid
            event = OpencodeEvent.from_dict(data)
            if event:
                self._handle_event(event)
        except json.JSONDecodeError:
            stripped = line.rstrip()
            if stripped:
                self.event_handler.on_thinking(stripped + "\n")

    def _handle_event(self, event: OpencodeEvent):
        """Handle a single parsed Opencode event."""
        if isinstance(event, TextEvent):
            self.stdout_lines.append(event.text)
            if self.event_handler:
                self.event_handler.on_thinking(event.text)

        elif isinstance(event, StepFinishEvent):
            self._update_usage_from_step(event)

        elif isinstance(event, ToolUseEvent):
            if event.status in ("success", "error"):
                args = _to_args_dict(event.input_data)
                stdout = str(event.output_data) if event.output_data is not None else ""

                if self.event_handler:
                    self.event_handler.on_tool_call(event.tool_name, args)
                    self.event_handler.on_tool_result(
                        tool=event.tool_name,
                        stdout=stdout,
                    )

    def _update_usage_from_step(self, event: StepFinishEvent) -> None:
        """Fold a step_finish payload into the running usage totals."""
        tokens: dict[str, Any] = event.tokens or {}
        cache: dict[str, Any] = tokens.get("cache") or {}
        cache_read = int(cache.get("read") or 0)
        cache_write = int(cache.get("write") or 0)
        cached = cache_read + cache_write
        step_usage = TokenUsage(
            input_tokens=int(tokens.get("input") or 0) + cached,
            output_tokens=int(tokens.get("output") or 0) + int(tokens.get("reasoning") or 0),
            cached_input_tokens=cached,
            turns=1,
        )
        self._accumulated_tokens = self._accumulated_tokens + step_usage
        if event.cost is not None:
            self._accumulated_cost_usd += float(event.cost)
            self._saw_cost = True
        self.usage = ProviderUsage(
            tokens=self._accumulated_tokens,
            total_cost_usd=self._accumulated_cost_usd if self._saw_cost else None,
            provider="opencode",
        )
        if self.event_handler is not None:
            on_usage = getattr(self.event_handler, "on_usage", None)
            if on_usage is not None:
                on_usage(
                    {
                        "input_tokens": step_usage.input_tokens,
                        "output_tokens": step_usage.output_tokens,
                        "cache_read_input_tokens": cache_read,
                        "cache_creation_input_tokens": cache_write,
                    }
                )


@register_provider("opencode")
class OpencodeCodingAgent(CLICodingAgent):
    """Coding agent implementation using the Opencode CLI tool."""

    def __init__(
        self,
        model: str | None = None,
        event_handler: AgentEventHandler | None = None,
        event_handlers: Iterable[AgentEventHandler] | None = None,
        mcp_servers: Sequence[object] | None = None,
        sandbox: bool | SandboxConfig = False,
        executor: CommandExecutor | None = None,
    ):
        """Initialize the Opencode coding agent.

        Args:
            model: Optional model name to use.
            event_handler: Optional event handler for UI updates.
            mcp_servers: Optional list of MCP server configurations.
            sandbox: Not supported for Opencode; must be False.
            executor: Optional command executor for binary lookup and process execution.

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
        super().__init__("opencode", model, event_handler, event_handlers, executor=executor)

    @property
    def _log_prefix(self) -> str:
        """Return the log prefix for this agent."""
        return "[Opencode]"

    def _get_command(self, prompt: str, resume_session_id: str | None = None) -> list[str]:
        cmd = [self.binary_path, "run"]

        if resume_session_id:
            cmd.extend(["--session", resume_session_id])

        cmd.append(f'"{prompt}"')

        if self.model:
            cmd.extend(["--model", self.model])

        cmd.extend(["--format=json"])

        return cmd

    def _create_session(
        self,
        cmd: list[str],
        cwd: str | None = None,
        timeout: int = 300,
        silent: bool = False,
        on_process_started: Callable[[CommandHandle], None] | None = None,
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
            event_handler=self.event_handler,
            executor=self.executor,
            on_process_started=on_process_started,
        )
