from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Callable
from typing import Any

from agentshim.trajectory import TrajectoryRecorderProtocol

from ..base import register_provider
from ..cli_agent import CLICodingAgent, CLIGenerationSession
from ..events import AgentEventHandler
from ..mcp_config import HttpMcpServer, McpServerConfig
from ..sandbox import SandboxConfig
from ..usage import ProviderUsage, TokenUsage
from .events import (
    CopilotEvent,
    ErrorEvent,
    IntentEvent,
    MessageDeltaEvent,
    MessageEvent,
    ResultEvent,
    SessionStartEvent,
    ToolResultEvent,
    ToolUseEvent,
    TurnEndEvent,
    UsageEvent,
)


class CopilotGenerationSession(CLIGenerationSession):
    """Session that parses Copilot CLI JSONL events."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.tool_map: dict[str, str] = {}
        self.tool_start_times: dict[str, float] = {}
        self.tool_args: dict[str, Any] = {}
        self.final_result: str | None = None
        self._seen_message_deltas: set[str] = set()
        self._streamed_text_chunks: list[str] = []
        self._accumulated_tokens = TokenUsage()
        self._turn_count = 0
        self._message_output_tokens = 0
        self._saw_usage_event = False

    def _refresh_usage(self) -> None:
        effective_output_tokens = self._accumulated_tokens.output_tokens
        if not self._saw_usage_event:
            effective_output_tokens = self._message_output_tokens
        self.usage = ProviderUsage(
            tokens=TokenUsage(
                input_tokens=self._accumulated_tokens.input_tokens,
                output_tokens=effective_output_tokens,
                cached_input_tokens=self._accumulated_tokens.cached_input_tokens,
                turns=self._turn_count,
            ),
            total_cost_usd=None,
            provider="copilot",
        )

    def _process_stdout(self, line: str) -> None:
        if not line:
            return
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            self.stdout_lines.append(line.rstrip())
            if not self.silent:
                if self._at_line_start:
                    self._log_raw(f"{self.log_prefix} ")
                self._log_raw(line.rstrip() + "\n")
                self._at_line_start = True
            return

        if not isinstance(data, dict):
            return

        event = CopilotEvent.from_dict(data)
        if event is None:
            return
        self._handle_event(event)

    def _handle_event(self, event: CopilotEvent) -> None:
        self._update_state(event)
        if not self.silent:
            self._render_event(event)

    def _update_state(self, event: CopilotEvent) -> None:
        if isinstance(event, SessionStartEvent):
            if self.session_id is None and event.session_id:
                self.session_id = event.session_id
            return

        if isinstance(event, MessageDeltaEvent):
            if event.message_id:
                self._seen_message_deltas.add(event.message_id)
            if event.delta_content:
                self._streamed_text_chunks.append(event.delta_content)
                if self.event_handler:
                    self.event_handler.on_thinking(event.delta_content)
            return

        if isinstance(event, MessageEvent):
            if event.content:
                self.stdout_lines.append(event.content)
                self.final_result = event.content
                if not self._saw_usage_event:
                    self._message_output_tokens += event.output_tokens
                    self._refresh_usage()
                if self.event_handler:
                    self.event_handler.on_thinking(event.content)
            return

        if isinstance(event, ToolUseEvent):
            if event.tool_id:
                self.tool_map[event.tool_id] = event.tool_name
                self.tool_start_times[event.tool_id] = time.time()
                self.tool_args[event.tool_id] = event.arguments
                if self.event_handler:
                    self.event_handler.on_tool_call(event.tool_name, event.arguments)
            return

        if isinstance(event, ToolResultEvent):
            if not event.tool_id:
                return
            event.tool_name_resolved = self.tool_map.get(event.tool_id, "Tool")
            start_time = self.tool_start_times.get(event.tool_id)
            duration = time.time() - start_time if start_time else None
            args = self.tool_args.get(event.tool_id, {})
            stdout = event.output or event.error_message

            self.recorder.add_tool_call(
                tool=event.tool_name_resolved,
                args=args,
                stdout=stdout,
                exit_code=event.exit_code,
                duration=duration,
            )
            if self.event_handler:
                self.event_handler.on_tool_result(
                    tool=event.tool_name_resolved,
                    stdout=stdout,
                    exit_code=event.exit_code,
                    duration=duration,
                )
            return

        if isinstance(event, UsageEvent):
            self._saw_usage_event = True
            cached = event.cache_read_tokens + event.cache_write_tokens
            self._accumulated_tokens = self._accumulated_tokens + TokenUsage(
                input_tokens=event.input_tokens + cached,
                output_tokens=event.output_tokens + event.reasoning_tokens,
                cached_input_tokens=cached,
                turns=0,
            )
            self._refresh_usage()
            return

        if isinstance(event, TurnEndEvent):
            self._turn_count += 1
            self._refresh_usage()
            return

        if isinstance(event, ResultEvent):
            if self.session_id is None and event.session_id:
                self.session_id = event.session_id
            return

        if isinstance(event, ErrorEvent):
            if event.message:
                self.stdout_lines.append(event.message)

    def _render_event(self, event: CopilotEvent) -> None:
        if isinstance(event, MessageDeltaEvent):
            if event.delta_content:
                self._print_stream_content(event.delta_content)
            return

        if isinstance(event, MessageEvent):
            if event.message_id and event.message_id in self._seen_message_deltas:
                return
            if event.content:
                self._print_stream_content(event.content)
            return

        if not self._at_line_start:
            self._log_raw("\n")
            self._at_line_start = True

        if isinstance(event, IntentEvent):
            output = event.render(self.log_prefix)
        else:
            output = event.render(self.log_prefix)
        if output:
            self._log_raw(output + "\n")

    def run(self, prompt: str) -> str:
        super().run(prompt)
        if self.final_result:
            return self.final_result
        if self._streamed_text_chunks:
            return "".join(self._streamed_text_chunks).strip()
        return "\n".join(self.stdout_lines)


@register_provider("copilot", "github-copilot", "copilot-cli")
class CopilotCodingAgent(CLICodingAgent):
    """Coding agent implementation using the GitHub Copilot CLI tool."""

    def __init__(
        self,
        model: str | None = None,
        recorder: TrajectoryRecorderProtocol | None = None,
        event_handler: AgentEventHandler | None = None,
        mcp_servers: list[McpServerConfig] | None = None,
        sandbox: bool | SandboxConfig = False,
    ):
        if sandbox:
            raise NotImplementedError("sandbox is not supported for CopilotCodingAgent")
        super().__init__("copilot", model, recorder, event_handler, mcp_servers)

    @property
    def copilot_path(self) -> str:
        """Return path to copilot binary (for backward compatibility)."""
        return self.binary_path

    @property
    def _log_prefix(self) -> str:
        return "[Copilot]"

    def _build_mcp_config_json(self) -> str:
        servers: dict[str, dict[str, Any]] = {}
        for server in self.mcp_servers:
            if isinstance(server, HttpMcpServer):
                entry: dict[str, Any] = {"type": "sse", "url": server.url}
                if server.headers:
                    entry["headers"] = dict(server.headers)
                servers[server.name] = entry
            else:
                entry = {"command": server.command, "args": server.args}
                if server.env:
                    entry["env"] = server.env
                servers[server.name] = entry
        return json.dumps({"mcpServers": servers})

    def _get_command(self, prompt: str, resume_session_id: str | None = None) -> list[str]:
        cmd = [
            self.binary_path,
            "--output-format",
            "json",
            "--stream",
            "off",
            "--allow-all-tools",
            "--allow-all-paths",
            "--allow-all-urls",
        ]
        if resume_session_id:
            cmd.extend(["--resume", resume_session_id])
        if self.model:
            cmd.extend(["--model", self.model])
        if self.mcp_servers:
            cmd.extend(["--additional-mcp-config", self._build_mcp_config_json()])
        cmd.extend(["-p", prompt])
        return cmd

    def _create_session(
        self,
        cmd: list[str],
        cwd: str | None = None,
        timeout: int = 300,
        silent: bool = False,
        recorder: TrajectoryRecorderProtocol | None = None,
        on_process_started: Callable[[subprocess.Popen[str]], None] | None = None,
    ) -> CopilotGenerationSession:
        return CopilotGenerationSession(
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
