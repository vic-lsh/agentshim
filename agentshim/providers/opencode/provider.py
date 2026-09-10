"""The opencode provider: argv, parser, MCP install, exit classification.

Flags verified against ``opencode run --help`` on opencode 1.2.18:

- ``run`` takes the message as a positional, but reads it from stdin when
  none is given, so the prompt never has to go in argv.
- ``-s, --session`` continues a stored session by id, which is the id every
  frame reports as ``sessionID``.
- ``-m, --model`` takes ``provider/model``; omitting it leaves the model
  configured in the user's own opencode config.
- ``--format`` accepts ``default`` or ``json``.
- ``--thinking`` makes the run publish its reasoning parts; without it
  opencode drops them from the stream entirely.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agentshim.core.errors import ProviderCapabilityError
from agentshim.core.mcp import (
    HttpMcpServer,
    NoopInstallation,
    StdioMcpServer,
    install_config_file,
)
from agentshim.core.profile import McpMechanism, OutputSchemaStyle, ProviderProfile

from .parser import OpencodeStreamParser

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from agentshim.core.errors import AgentShimError, CliExitError
    from agentshim.core.events import AgentEvent
    from agentshim.core.mcp import McpServer
    from agentshim.core.provider import ArgvContext, McpInstallation

MCP_CONFIG_FILENAME = "opencode.json"
MCP_SERVER_KEY = "mcp"

#: Top-level keys the provider needs when it creates the config file itself.
MCP_CONFIG_DEFAULTS = {"$schema": "https://opencode.ai/config.json"}

_NO_WORKSPACE = "opencode installs MCP servers into <cwd>/opencode.json; the turn needs a cwd"

PROFILE = ProviderProfile(
    name="opencode",
    display_name="opencode",
    binary="opencode",
    supports_resume=True,
    # ``run --session <id>`` continues a stored session.
    supports_reasoning_effort=False,
    # ``--variant`` is a per-model knob, not a portable effort level.
    mcp=McpMechanism.CONFIG_FILE,
    output_schema=OutputSchemaStyle.NONE,
    schema_dialect=None,
    state_dirs=(".local/share/opencode", ".config/opencode"),
    # opencode follows the XDG layout on macOS too.
    darwin_state_dirs=(),
    # opencode authenticates through ``opencode auth``, which writes
    # auth.json into its own state directory; there is no API-key variable.
    auth_env_vars=(),
    skill_dirs=(".opencode/skills",),
    container_install=(
        "apt-get update && apt-get install -y --no-install-recommends curl ca-certificates",
        "curl -fsSL https://opencode.ai/install | bash",
        # The installer picks one of two prefixes depending on the image.
        "ln -sf /root/.opencode/bin/opencode /usr/local/bin/opencode 2>/dev/null || "
        "ln -sf /root/.local/bin/opencode /usr/local/bin/opencode",
    ),
)


class OpencodeProvider:
    """opencode's non-interactive ``run`` mode with JSON events."""

    profile = PROFILE

    def build_argv(self, ctx: ArgvContext) -> list[str]:
        """Build the argv for one turn. The prompt is never part of it."""
        argv = [ctx.binary_path, "run"]
        if ctx.resume_session_id:
            argv += ["--session", ctx.resume_session_id]
        argv += ["--format", "json", "--thinking"]
        if ctx.model:
            argv += ["--model", ctx.model]
        argv += list(ctx.mcp_argv)
        argv += list(ctx.extra_args)
        return argv

    def new_parser(
        self,
        emit: Callable[[AgentEvent], None],
        *,
        expect_structured: bool,
    ) -> OpencodeStreamParser:
        """Build the stream parser for one run."""
        return OpencodeStreamParser(emit, expect_structured=expect_structured)

    def install_mcp(self, workspace: Path | None, servers: Sequence[McpServer]) -> McpInstallation:
        """Merge ``servers`` into ``<workspace>/opencode.json``."""
        if not servers:
            return NoopInstallation()
        if workspace is None:
            raise ProviderCapabilityError(_NO_WORKSPACE)
        return install_config_file(
            workspace / MCP_CONFIG_FILENAME,
            server_key=MCP_SERVER_KEY,
            servers={server.name: mcp_entry(server) for server in servers},
            defaults=MCP_CONFIG_DEFAULTS,
        )

    def classify_exit(self, error: CliExitError, *, resumed: bool) -> AgentShimError:
        """Return the exit error unchanged.

        ``opencode run --session`` on an unknown id fails the same way as any
        other run, with no distinguishable exit code, so mapping a resumed
        turn's failure to ``SessionResumeError`` would be a guess.
        """
        _ = resumed
        return error


def mcp_entry(server: McpServer) -> dict[str, Any]:
    """Render one server the way ``opencode.json`` describes it.

    opencode validates the block strictly: a local server takes one combined
    ``command`` array and calls its environment ``environment``. Its
    non-interactive ``run`` mode auto-answers permission prompts, so no
    extra ``permission`` block is needed.
    """
    if isinstance(server, HttpMcpServer):
        entry: dict[str, Any] = {"type": "remote", "url": server.url, "enabled": True}
        if server.headers:
            entry["headers"] = dict(server.headers)
        return entry
    stdio: StdioMcpServer = server
    rendered: dict[str, Any] = {
        "type": "local",
        "command": [stdio.command, *stdio.args],
        "enabled": True,
    }
    if stdio.env:
        rendered["environment"] = dict(stdio.env)
    return rendered
