"""The Claude Code provider: argv, parser, MCP install, exit classification."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from agentshim.core.errors import ProviderCapabilityError, SessionResumeError
from agentshim.core.mcp import HttpMcpServer, NoopInstallation, StdioMcpServer, install_config_file
from agentshim.core.profile import McpMechanism, OutputSchemaStyle, ProviderProfile, SchemaDialect

from .parser import ClaudeStreamParser
from .sandbox import SANDBOX_ENV, SandboxConfig, build_settings, resolve_sandbox

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from agentshim.core.errors import AgentShimError, CliExitError
    from agentshim.core.events import AgentEvent
    from agentshim.core.mcp import McpServer
    from agentshim.core.provider import ArgvContext, McpInstallation

MCP_CONFIG_FILENAME = ".mcp.json"
MCP_SERVER_KEY = "mcpServers"

PROFILE = ProviderProfile(
    name="claude",
    display_name="Claude Code",
    binary="claude",
    supports_resume=True,
    supports_reasoning_effort=True,
    mcp=McpMechanism.CONFIG_FILE,
    output_schema=OutputSchemaStyle.INLINE_JSON,
    # ``--json-schema`` accepts open-ended object maps, unlike Codex's
    # ``--output-schema`` subset, so ``dict[str, T]`` fields stay native.
    schema_dialect=SchemaDialect.OPEN,
    state_dirs=(".claude", ".claude.json", ".config/claude"),
    darwin_state_dirs=("Library/Application Support/claude", "Library/Caches/claude"),
    auth_env_vars=(
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_CUSTOM_HEADERS",
    ),
    skill_dirs=(".claude/skills",),
    container_install=(
        "apt-get update && apt-get install -y --no-install-recommends curl ca-certificates",
        "curl -fsSL https://claude.ai/install.sh | bash",
        # The installer drops the binary in /root/.local/bin.
        "ln -sf /root/.local/bin/claude /usr/local/bin/claude",
    ),
)


class ClaudeProvider:
    """Claude Code. ``sandbox`` is a provider option, not a portable one."""

    profile = PROFILE

    def __init__(self, *, sandbox: bool | SandboxConfig | None = None) -> None:
        """Fix the sandbox option for every turn this provider runs.

        ``True`` takes the default config; ``None`` and ``False`` both mean
        unsandboxed.
        """
        self.sandbox: SandboxConfig | None = resolve_sandbox(sandbox)

    @property
    def sandbox_env(self) -> dict[str, str]:
        """Environment a sandboxed run needs; empty when no sandbox is set.

        Merge this into ``CliAgent(env=...)`` when you enable the sandbox.
        """
        return dict(SANDBOX_ENV) if self.sandbox is not None else {}

    def build_argv(self, ctx: ArgvContext) -> list[str]:
        """Build the print-mode argv; the prompt is not in it, it goes on stdin."""
        argv = [ctx.binary_path]
        if ctx.resume_session_id:
            # Before -p: claude parses the resume target as a session flag,
            # not as part of the print-mode invocation.
            argv += ["--resume", ctx.resume_session_id]
        argv += [
            "-p",
            "--dangerously-skip-permissions",
            "--output-format",
            "stream-json",
            "--verbose",
        ]
        if ctx.model:
            argv += ["--model", ctx.model]
        if ctx.reasoning_effort:
            argv += ["--effort", ctx.reasoning_effort]
        if ctx.schema_inline:
            argv += ["--json-schema", ctx.schema_inline]
        if self.sandbox is not None:
            argv += ["--settings", json.dumps(build_settings(self.sandbox))]
        argv += list(ctx.mcp_argv)
        argv += list(ctx.extra_args)
        return argv

    def new_parser(
        self,
        emit: Callable[[AgentEvent], None],
        *,
        expect_structured: bool,
    ) -> ClaudeStreamParser:
        """Return a parser for one run's ``stream-json`` output."""
        return ClaudeStreamParser(emit, expect_structured=expect_structured)

    def install_mcp(self, workspace: Path | None, servers: Sequence[McpServer]) -> McpInstallation:
        """Write the servers into ``<workspace>/.mcp.json`` for the turn's lifetime."""
        if not servers:
            return NoopInstallation()
        if workspace is None:
            msg = "claude installs MCP servers into <cwd>/.mcp.json; the turn needs a cwd"
            raise ProviderCapabilityError(msg)
        return install_config_file(
            workspace / MCP_CONFIG_FILENAME,
            server_key=MCP_SERVER_KEY,
            servers={server.name: mcp_entry(server) for server in servers},
        )

    def classify_exit(self, error: CliExitError, *, resumed: bool) -> AgentShimError:
        """Any nonzero exit on a resumed turn means the conversation is gone.

        ``claude --resume`` gives no distinguishable exit code for a missing
        transcript, and a resumed turn that fails is unusable either way:
        the caller has to start a fresh conversation.
        """
        if resumed:
            session_id = _resumed_session_id(error.argv)
            if session_id is not None:
                return SessionResumeError(
                    error.argv, error.returncode, session_id, error.stdout, error.stderr
                )
        return error


def _resumed_session_id(argv: Sequence[str]) -> str | None:
    args = list(argv)
    if "--resume" not in args:
        return None
    index = args.index("--resume")
    if index + 1 >= len(args):
        return None
    return args[index + 1]


def mcp_entry(server: McpServer) -> dict[str, Any]:
    """Render one server the way ``.mcp.json`` describes it."""
    if isinstance(server, HttpMcpServer):
        # Claude validates the config strictly and needs an explicit type.
        entry: dict[str, Any] = {"type": "sse", "url": server.url}
        if server.headers:
            entry["headers"] = dict(server.headers)
        return entry
    stdio: StdioMcpServer = server
    rendered: dict[str, Any] = {"command": stdio.command, "args": list(stdio.args)}
    if stdio.env:
        rendered["env"] = dict(stdio.env)
    return rendered
