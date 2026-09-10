"""The Gemini CLI provider: argv, parser, MCP install, exit classification.

Flags verified against ``gemini --help`` on Gemini CLI 0.26.0
(``@google/gemini-cli@0.26.0``):

- ``-y, --yolo`` automatically approves every tool call.
- ``-o, --output-format`` accepts ``text``, ``json`` and ``stream-json``.
- ``-m, --model`` names the model; omitting it leaves the CLI's own default.
- ``-r, --resume`` takes ``latest``, a 1-based index, or a session id.
  ``SessionSelector.findSession`` matches the id first, and that id is the
  one the ``init`` frame reports, so resuming by captured id is exact.

The prompt is piped on stdin: ``gemini`` reads stdin whenever it is not a
TTY and uses it as the input, so nothing has to go in argv.
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

from .parser import GeminiStreamParser

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from agentshim.core.errors import AgentShimError, CliExitError
    from agentshim.core.events import AgentEvent
    from agentshim.core.mcp import McpServer
    from agentshim.core.provider import ArgvContext, McpInstallation

MCP_CONFIG_PATH = (".gemini", "settings.json")
MCP_SERVER_KEY = "mcpServers"

_NO_WORKSPACE = "gemini installs MCP servers into <cwd>/.gemini/settings.json; the turn needs a cwd"

PROFILE = ProviderProfile(
    name="gemini",
    display_name="Gemini CLI",
    binary="gemini",
    supports_resume=True,
    # ``--resume`` restores a stored session by id.
    supports_reasoning_effort=False,
    # No effort or thinking-budget flag exists on the CLI.
    mcp=McpMechanism.CONFIG_FILE,
    output_schema=OutputSchemaStyle.NONE,
    schema_dialect=None,
    state_dirs=(".gemini", ".config/gemini"),
    # Gemini stores everything under those two on macOS as well.
    darwin_state_dirs=(),
    auth_env_vars=("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    skill_dirs=(".gemini/skills",),
    container_install=(
        "apt-get update && apt-get install -y --no-install-recommends curl ca-certificates",
        # nodejs.org stays reachable from hosts where the Debian mirrors time
        # out, so the tarball is more reliable here than apt-get install nodejs.
        "command -v node >/dev/null || { set -e; V=v20.18.1; A=linux-x64; cd /tmp && "
        'curl -fsSL --retry 5 --retry-delay 5 -o node.tgz "https://nodejs.org/dist/$V/node-$V-$A.tar.gz" && '
        "mkdir -p /opt/node && tar -xzf node.tgz -C /opt/node --strip-components=1 && "
        "ln -sf /opt/node/bin/node /usr/local/bin/node && ln -sf /opt/node/bin/npm /usr/local/bin/npm && "
        "ln -sf /opt/node/bin/npx /usr/local/bin/npx && /opt/node/bin/npm config set prefix /usr/local && "
        "rm -f node.tgz; }",
        "npm install -g @google/gemini-cli",
    ),
)


class GeminiProvider:
    """Gemini CLI in non-interactive ``stream-json`` mode."""

    profile = PROFILE

    def build_argv(self, ctx: ArgvContext) -> list[str]:
        """Build the argv for one turn. The prompt is never part of it."""
        argv = [ctx.binary_path, "--yolo", "--output-format", "stream-json"]
        if ctx.model:
            argv += ["--model", ctx.model]
        if ctx.resume_session_id:
            argv += ["--resume", ctx.resume_session_id]
        argv += list(ctx.mcp_argv)
        argv += list(ctx.extra_args)
        return argv

    def new_parser(
        self,
        emit: Callable[[AgentEvent], None],
        *,
        expect_structured: bool,
    ) -> GeminiStreamParser:
        """Build the stream parser for one run."""
        return GeminiStreamParser(emit, expect_structured=expect_structured)

    def install_mcp(self, workspace: Path | None, servers: Sequence[McpServer]) -> McpInstallation:
        """Merge ``servers`` into ``<workspace>/.gemini/settings.json``."""
        if not servers:
            return NoopInstallation()
        if workspace is None:
            raise ProviderCapabilityError(_NO_WORKSPACE)
        return install_config_file(
            workspace.joinpath(*MCP_CONFIG_PATH),
            server_key=MCP_SERVER_KEY,
            servers={server.name: mcp_entry(server) for server in servers},
        )

    def classify_exit(self, error: CliExitError, *, resumed: bool) -> AgentShimError:
        """Return the exit error unchanged.

        Gemini CLI reports a refused resume the same way it reports any other
        startup failure: a fatal error on stderr and a generic nonzero exit.
        Mapping a resumed turn's failure to ``SessionResumeFailed`` would
        therefore be a guess, so ``resumed`` does not change the outcome.
        """
        _ = resumed
        return error


def mcp_entry(server: McpServer) -> dict[str, Any]:
    """Render one server the way ``.gemini/settings.json`` describes it.

    ``trust: true`` skips Gemini's per-tool approval prompt for servers the
    caller installed on purpose.
    """
    if isinstance(server, HttpMcpServer):
        # ``httpUrl`` selects streamable HTTP; ``url`` would mean SSE.
        entry: dict[str, Any] = {"httpUrl": server.url, "trust": True}
        if server.headers:
            entry["headers"] = dict(server.headers)
        return entry
    stdio: StdioMcpServer = server
    rendered: dict[str, Any] = {"command": stdio.command, "args": list(stdio.args), "trust": True}
    if stdio.env:
        rendered["env"] = dict(stdio.env)
    return rendered
