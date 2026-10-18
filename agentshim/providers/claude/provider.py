"""The Claude Code provider: argv, parser, MCP install, exit classification."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from agentshim.core.auth import (
    CredentialKind,
    EndpointSpec,
    EnvCredential,
    ExternalCredential,
    FileCredential,
    SettingsSpec,
)
from agentshim.core.errors import FailureKind, ProviderCapabilityError, SessionResumeError
from agentshim.core.mcp import HttpMcpServer, NoopInstallation, StdioMcpServer, install_config_file
from agentshim.core.profile import (
    ConfigScope,
    McpMechanism,
    McpScope,
    OutputSchemaStyle,
    ProviderProfile,
    SchemaDialect,
    SkillScope,
    SkillSignal,
)

from .parser import ClaudeStreamParser
from .sandbox import SANDBOX_ENV, SandboxConfig, build_settings, resolve_sandbox
from .tools import ClaudeToolPolicy, resolve_tools
from .user_hooks import ClaudeHook, resolve_hooks

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from agentshim.core.errors import AgentShimError, CliExitError
    from agentshim.core.events import AgentEvent
    from agentshim.core.mcp import McpServer
    from agentshim.core.provider import ArgvContext, McpInstallation

MCP_CONFIG_FILENAME = ".mcp.json"
MCP_SERVER_KEY = "mcpServers"

#: ``SkillScope.PROJECT``: load settings only from the workspace's
#: ``.claude/settings.json`` and ``.claude/settings.local.json`` (plus
#: ``--settings`` and managed policy, which always apply). Leaving out the
#: ``user`` source drops ``~/.claude/skills``, the user's plugins (with their
#: skills, hooks and MCP servers), ``~/.claude/CLAUDE.md`` and the user's
#: settings (default model, permissions, hooks, ``env``, ``apiKeyHelper``).
#: Credentials are not settings: OAuth in ``.credentials.json`` and the
#: ``auth_env_vars`` keep working. Claude Code's built-in skills and claude.ai
#: account connectors are not user settings and stay.
PROJECT_SETTING_SOURCES = ("--setting-sources", "project,local")

#: ``ConfigScope.PROJECT``: the same ``--setting-sources`` drops the user's
#: settings (hooks, ``env``, permissions, default model), ``~/.claude/CLAUDE.md``
#: and plugins; auto-memory is not a setting source and is switched off in the
#: turn's ``--settings`` instead, so ``~/.claude/projects/*/memory`` is neither
#: read nor written. Built-in skills, the workspace's ``CLAUDE.md``,
#: ``.claude/`` settings and skills, and credentials keep working.
NO_AUTO_MEMORY_SETTINGS = {"autoMemoryEnabled": False}

#: ``McpScope.SESSION``: ``--strict-mcp-config`` makes ``--mcp-config`` the
#: only source of MCP servers. It drops the user's and the project's
#: (``.mcp.json``) servers, plugin servers and claude.ai account connectors,
#: so the servers the turn was given are passed inline as ``--mcp-config``.
STRICT_MCP_FLAG = "--strict-mcp-config"

PROFILE = ProviderProfile(
    name="claude",
    # Model ids Claude Code's ``--model`` takes and agentshim has run, the
    # default first. Context windows are in ``agentshim.models``.
    models=(
        "claude-opus-5",
        "claude-opus-5-5",
        "claude-fable-5-1",
        "claude-fable-5",
        "claude-sonnet-5-5",
        "claude-sonnet-5",
        "claude-haiku-5-5",
        "claude-haiku-4-5",
    ),
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
        "CLAUDE_CODE_OAUTH_TOKEN",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_CUSTOM_HEADERS",
    ),
    skill_dirs=(".claude/skills",),
    # ``system/init`` lists ``skills``; the ``Skill`` tool call names one.
    skill_discovery=SkillSignal.STRUCTURED,
    skill_invocation=SkillSignal.STRUCTURED,
    skill_scopes=frozenset({SkillScope.ALL, SkillScope.PROJECT}),
    mcp_scopes=frozenset({McpScope.ALL, McpScope.SESSION}),
    config_scopes=frozenset({ConfigScope.ALL, ConfigScope.PROJECT}),
    container_install=(
        "apt-get update && apt-get install -y --no-install-recommends curl ca-certificates",
        "curl -fsSL https://claude.ai/install.sh | bash",
        # The installer drops the binary in /root/.local/bin.
        "ln -sf /root/.local/bin/claude /usr/local/bin/claude",
    ),
    # Claude Code refuses --dangerously-skip-permissions as root unless this
    # is set.
    container_env={"IS_SANDBOX": "1"},
    # Documented Claude Code variable that relocates ~/.claude.
    state_root_env="CLAUDE_CONFIG_DIR",
    auth_files=(
        ".claude/.credentials.json",
        ".claude/settings.json",
        ".claude/settings.local.json",
        ".claude.json",
    ),
    credential_files=(".claude/.credentials.json",),
    # Precedence as Claude Code documents it ("Authentication precedence",
    # code.claude.com/docs/en/iam): a cloud provider selected by variable,
    # then a bearer token, an API key, the ``apiKeyHelper`` setting, a
    # subscription token (``claude setup-token``), an Anthropic profile or
    # federation credential, and last the stored Claude.ai login. Checked
    # against 2.1.300: ``apiKeyHelper`` beats CLAUDE_CODE_OAUTH_TOKEN.
    credential_sources=(
        ExternalCredential(
            "a cloud provider",
            env=("CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX", "CLAUDE_CODE_USE_FOUNDRY"),
        ),
        EnvCredential("ANTHROPIC_AUTH_TOKEN", CredentialKind.API_KEY),
        EnvCredential("ANTHROPIC_API_KEY", CredentialKind.API_KEY),
        ExternalCredential("an apiKeyHelper command", settings_keys=("apiKeyHelper",)),
        EnvCredential("CLAUDE_CODE_OAUTH_TOKEN", CredentialKind.SUBSCRIPTION),
        ExternalCredential(
            "an Anthropic profile or federation credential",
            env=("ANTHROPIC_PROFILE", "ANTHROPIC_FEDERATION_RULE_ID"),
            paths=(".config/anthropic/configs", ".config/anthropic/active_config"),
        ),
        FileCredential(".claude/.credentials.json", kind=CredentialKind.SUBSCRIPTION),
    ),
    endpoint=EndpointSpec("ANTHROPIC_BASE_URL", companion_env=("ANTHROPIC_CUSTOM_HEADERS",)),
    # User settings; the ``env`` block overrides the process environment
    # (checked against 2.1.300: a settings ANTHROPIC_BASE_URL beats an
    # exported one).
    settings=SettingsSpec((".claude/settings.json", ".claude/settings.local.json"), env_key="env"),
    mcp_config_file=MCP_CONFIG_FILENAME,
)


class ClaudeProvider:
    """Claude Code. ``sandbox``, ``hooks`` and ``tools`` are provider options, not portable ones."""

    profile = PROFILE

    def __init__(
        self,
        *,
        sandbox: bool | SandboxConfig | None = None,
        hooks: Sequence[ClaudeHook] = (),
        tools: ClaudeToolPolicy | None = None,
    ) -> None:
        """Fix the sandbox, hooks and tool policy for every turn this provider runs.

        ``sandbox=True`` takes the default config; ``None`` and ``False`` both
        mean unsandboxed. ``hooks`` are added to the settings Claude Code
        loads for the turn, after agentshim's own read-confinement hook.
        ``tools=None`` runs every turn with ``--dangerously-skip-permissions``
        (all built-in tools, every call allowed); a ``ClaudeToolPolicy`` runs
        it with only the tools and allow rules the policy names, under
        ``--permission-mode dontAsk``.
        """
        self.sandbox: SandboxConfig | None = resolve_sandbox(sandbox)
        self.hooks: tuple[ClaudeHook, ...] = resolve_hooks(hooks)
        self.tools: ClaudeToolPolicy | None = resolve_tools(tools)

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
        argv.append("-p")
        if self.tools is None:
            argv.append("--dangerously-skip-permissions")
        else:
            argv += self.tools.argv(structured_output=ctx.schema_inline is not None)
        argv += ["--output-format", "stream-json", "--verbose"]
        isolate_config = ctx.config_scope is ConfigScope.PROJECT
        if ctx.skill_scope is SkillScope.PROJECT or isolate_config:
            argv += PROJECT_SETTING_SOURCES
        if ctx.model:
            argv += ["--model", ctx.model]
        if ctx.reasoning_effort:
            argv += ["--effort", ctx.reasoning_effort]
        if ctx.schema_inline:
            argv += ["--json-schema", ctx.schema_inline]
        settings = build_settings(self.sandbox, hooks=self.hooks)
        if isolate_config:
            settings.update(NO_AUTO_MEMORY_SETTINGS)
        if settings:
            argv += ["--settings", json.dumps(settings)]
        if ctx.mcp_scope is McpScope.SESSION:
            argv += _session_mcp_argv(ctx.mcp_servers)
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
        """A resumed turn that failed for no reason the stream named lost its conversation.

        ``claude --resume`` gives no distinguishable exit code for a missing
        transcript, so a resumed turn whose failure the parser could not
        classify is treated as a dead conversation. A failure the stream did
        classify (an overload, a usage limit, a login problem) happened inside
        a conversation that resumed, so it keeps its own kind.
        """
        if resumed and error.kind is FailureKind.OTHER:
            session_id = _resumed_session_id(error.argv)
            if session_id is not None:
                return SessionResumeError(
                    error.argv,
                    error.returncode,
                    session_id,
                    error.stdout,
                    error.stderr,
                    detail=error.detail,
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


def _session_mcp_argv(servers: Sequence[McpServer]) -> list[str]:
    """Flags that leave Claude Code exactly *servers* and nothing else."""
    argv = [STRICT_MCP_FLAG]
    if servers:
        config = {MCP_SERVER_KEY: {server.name: mcp_entry(server) for server in servers}}
        argv += ["--mcp-config", json.dumps(config)]
    return argv


def mcp_entry(server: McpServer) -> dict[str, Any]:
    """Render one server the way ``.mcp.json`` describes it."""
    if server.startup_timeout_s is not None:
        msg = "claude cannot configure a per-server MCP startup timeout"
        raise ProviderCapabilityError(msg)
    if server.tool_timeout_s is not None:
        msg = "claude cannot configure a per-server MCP tool timeout"
        raise ProviderCapabilityError(msg)
    if isinstance(server, HttpMcpServer):
        # Claude validates the config strictly and needs an explicit type; it
        # names the two remote transports exactly as the spec does.
        entry: dict[str, Any] = {"type": server.transport, "url": server.url}
        if server.headers:
            entry["headers"] = dict(server.headers)
        return entry
    stdio: StdioMcpServer = server
    rendered: dict[str, Any] = {"command": stdio.command, "args": list(stdio.args)}
    if stdio.env:
        rendered["env"] = dict(stdio.env)
    return rendered
