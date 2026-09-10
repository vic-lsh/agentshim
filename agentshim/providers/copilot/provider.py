"""The Copilot CLI provider: argv, parser, MCP install, exit classification."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from agentshim.core.mcp import FlagsInstallation, HttpMcpServer, NoopInstallation
from agentshim.core.profile import McpMechanism, OutputSchemaStyle, ProviderProfile

from .parser import CopilotStreamParser

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

    from agentshim.core.errors import AgentShimError, CliExitError
    from agentshim.core.events import AgentEvent
    from agentshim.core.mcp import McpServer, StdioMcpServer
    from agentshim.core.provider import ArgvContext, McpInstallation

MCP_CONFIG_FLAG = "--additional-mcp-config"
MCP_SERVER_KEY = "mcpServers"

PROFILE = ProviderProfile(
    name="copilot",
    display_name="Copilot CLI",
    binary="copilot",
    supports_resume=True,
    # ``--effort`` exists on the CLI but the provider does not expose it: the
    # levels are model-specific and unvalidated, so a request naming one is
    # rejected by the session instead of being silently mistranslated.
    supports_reasoning_effort=False,
    mcp=McpMechanism.CLI_FLAGS,
    output_schema=OutputSchemaStyle.NONE,
    schema_dialect=None,
    # ``COPILOT_HOME`` defaults to ``$HOME/.copilot`` (``copilot --help``,
    # ``copilot help environment``); ``~/.config/github-copilot`` holds the
    # GitHub Copilot credentials shared with the editor extensions.
    state_dirs=(".copilot", ".config/github-copilot"),
    # Copilot keeps no separate macOS state; ``.copilot`` is used there too.
    darwin_state_dirs=(),
    # ``copilot help environment``, in the CLI's own precedence order.
    auth_env_vars=("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"),
    # The CLI also reads ``.agents/skills`` and ``.claude/skills`` from the
    # workspace; ``.github/skills`` is its own convention.
    skill_dirs=(".github/skills",),
    # The npm package ships the CLI; the image supplies node and npm.
    container_install=("npm install -g @github/copilot",),
    # ``COPILOT_HOME`` relocates ~/.copilot; documented above and by
    # ``copilot help environment``.
    state_root_env="COPILOT_HOME",
    # Copilot documents no specific credential filenames. ``.copilot/`` on a
    # logged-in install holds ``config.json`` (opaque, not plain JSON; most
    # likely where the login state lives) and ``settings.json`` (user
    # config, e.g. the chosen model); ``.config/github-copilot/`` on the same
    # install held only ``versions.json``, which is not credential-bearing,
    # so it is left out.
    auth_files=(".copilot/config.json", ".copilot/settings.json"),
    # CLI_FLAGS: Copilot takes MCP servers as one --additional-mcp-config
    # flag, never a file.
    mcp_config_file=None,
)


class CopilotProvider:
    """GitHub Copilot CLI."""

    profile = PROFILE

    def build_argv(self, ctx: ArgvContext) -> list[str]:
        """Build the non-interactive argv for one turn.

        The prompt is not in it: ``copilot`` reads the prompt from stdin when
        stdin is not a terminal, which is what the session gives it, so the
        ``-p`` flag is not used. ``--stream off`` keeps the CLI to whole
        ``assistant.message`` frames, and the three ``--allow-all-*`` flags
        are what makes a non-interactive run answer its own permission
        prompts.
        """
        argv = [
            ctx.binary_path,
            "--output-format",
            "json",
            "--stream",
            "off",
            "--allow-all-tools",
            "--allow-all-paths",
            "--allow-all-urls",
        ]
        if ctx.resume_session_id:
            argv += ["--resume", ctx.resume_session_id]
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
    ) -> CopilotStreamParser:
        """Return a parser for one run.

        Copilot has no output-schema mode, so the parser records the flag and
        never acts on it.
        """
        return CopilotStreamParser(emit, expect_structured=expect_structured)

    def install_mcp(self, workspace: Path | None, servers: Sequence[McpServer]) -> McpInstallation:
        """Render the servers into one ``--additional-mcp-config`` flag.

        Copilot takes MCP servers as a JSON string on the command line, so
        nothing is written to the workspace and nothing needs restoring.
        """
        del workspace
        if not servers:
            return NoopInstallation()
        config = {MCP_SERVER_KEY: {server.name: mcp_entry(server) for server in servers}}
        return FlagsInstallation([MCP_CONFIG_FLAG, json.dumps(config)])

    def classify_exit(self, error: CliExitError, *, resumed: bool) -> AgentShimError:
        """Return the exit error unchanged.

        ``copilot --resume`` reports a missing session the same way it
        reports any other failure, and the id is not recoverable from the
        exit alone, so there is nothing to add to the error.
        """
        del resumed
        return error


def mcp_entry(server: McpServer) -> dict[str, Any]:
    """Render one server the way ``--additional-mcp-config`` describes it."""
    if isinstance(server, HttpMcpServer):
        # Copilot uses the spec's own transport names in ``type``.
        entry: dict[str, Any] = {"type": server.transport, "url": server.url}
        if server.headers:
            entry["headers"] = dict(server.headers)
        return entry
    stdio: StdioMcpServer = server
    rendered: dict[str, Any] = {"command": stdio.command, "args": list(stdio.args)}
    if stdio.env:
        rendered["env"] = dict(stdio.env)
    return rendered


def parse_mcp_servers(argv: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Recover the MCP servers rendered into *argv* by ``install_mcp``.

    The inverse of ``mcp_entry``, kept next to it so the two cannot drift:
    reads the ``--additional-mcp-config`` flag's JSON payload back and maps
    each entry to the canonical shape ``installed_mcp_servers`` returns.

    Args:
        argv: The argv a turn actually ran, as recorded on a ``CommandRequest``.

    Returns:
        One canonical entry per server, keyed by server name.
    """
    args = list(argv)
    if MCP_CONFIG_FLAG not in args:
        return {}
    index = args.index(MCP_CONFIG_FLAG)
    if index + 1 >= len(args):
        return {}
    config = json.loads(args[index + 1])
    servers = config.get(MCP_SERVER_KEY, {})
    return {name: _canonical_entry(raw) for name, raw in servers.items()}


def _canonical_entry(raw: Mapping[str, Any]) -> dict[str, Any]:
    if "url" in raw:
        return {"url": raw["url"], "transport": raw.get("type", "http")}
    return {
        "command": raw.get("command", ""),
        "args": list(raw.get("args", ())),
        "env": dict(raw.get("env", {})),
    }
