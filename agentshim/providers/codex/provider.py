"""The Codex provider: argv, parser, MCP install, exit classification."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from agentshim.core.errors import ProviderCapabilityError, SessionResumeError
from agentshim.core.mcp import FlagsInstallation, HttpMcpServer, NoopInstallation
from agentshim.core.profile import (
    McpMechanism,
    OutputSchemaStyle,
    ProviderProfile,
    SchemaDialect,
)

from .parser import CodexStreamParser

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

    from agentshim.core.errors import AgentShimError, CliExitError
    from agentshim.core.events import AgentEvent
    from agentshim.core.mcp import McpServer
    from agentshim.core.provider import ArgvContext, McpInstallation

#: The CLI release the container recipe installs, matching the host pin.
CLI_VERSION = "0.144.4"

#: Node is not in the Debian base image and apt is unreliable from some
#: hosts, so the tarball from nodejs.org is what the recipe fetches.
_NODE_INSTALL = (
    "command -v node >/dev/null || { set -e; "
    "V=v20.18.1; A=linux-x64; "
    "cd /tmp && "
    'curl -fsSL --retry 5 --retry-delay 5 -o node.tgz "https://nodejs.org/dist/$V/node-$V-$A.tar.gz" && '
    "mkdir -p /opt/node && "
    "tar -xzf node.tgz -C /opt/node --strip-components=1 && "
    "ln -sf /opt/node/bin/node /usr/local/bin/node && "
    "ln -sf /opt/node/bin/npm /usr/local/bin/npm && "
    "ln -sf /opt/node/bin/npx /usr/local/bin/npx && "
    "/opt/node/bin/npm config set prefix /usr/local && "
    "rm -f node.tgz; }"
)

#: Codex ships its linux-x64 native binary as an optional dependency, which
#: ``npm install -g`` skips under some npm configurations.
_CODEX_INSTALL = f"npm install -g --include=optional @openai/codex@{CLI_VERSION}"

_HTTP_HEADERS_UNSUPPORTED = (
    "codex configures MCP servers through --config overrides, which carry no HTTP headers"
)

PROFILE = ProviderProfile(
    name="codex",
    display_name="Codex",
    binary="codex",
    supports_resume=True,
    supports_reasoning_effort=True,
    mcp=McpMechanism.CLI_FLAGS,
    output_schema=OutputSchemaStyle.FILE_PATH,
    # ``--output-schema`` accepts only closed objects: every property is
    # declared and undeclared keys are forbidden.
    schema_dialect=SchemaDialect.STRICT,
    state_dirs=(".codex", ".config/codex"),
    darwin_state_dirs=(
        "Library/Application Support/codex",
        "Library/Application Support/com.openai.codex",
        "Library/Caches/codex",
    ),
    auth_env_vars=("OPENAI_API_KEY", "OPENAI_BASE_URL"),
    skill_dirs=(".agents/skills",),
    container_install=(_NODE_INSTALL, _CODEX_INSTALL),
    # Documented Codex CLI variable that relocates ~/.codex (``codex --help``:
    # "Layer $CODEX_HOME/<name>.config.toml on top of the base user config";
    # "auth still uses CODEX_HOME").
    state_root_env="CODEX_HOME",
    auth_files=(".codex/auth.json", ".codex/config.toml"),
    # CLI_FLAGS: Codex takes MCP servers as --config overrides, never a file.
    mcp_config_file=None,
)


class CodexProvider:
    """Codex (``codex exec --json``)."""

    profile = PROFILE

    def build_argv(self, ctx: ArgvContext) -> list[str]:
        """Build the ``codex exec`` command line for one turn.

        ``codex exec resume`` does not fall back to stdin when the prompt
        positional is omitted: only the literal ``-`` sentinel makes it read
        from stdin, and it has to sit right after the thread id. Plain
        ``codex exec`` treats stdin as the default, so it needs no sentinel.
        """
        argv = [ctx.binary_path, "exec"]
        if ctx.resume_session_id:
            argv += ["resume", ctx.resume_session_id, "-"]
        argv += ["--dangerously-bypass-approvals-and-sandbox", "--skip-git-repo-check", "--json"]
        if ctx.model:
            argv += ["--model", ctx.model]
        argv += _shell_path_config(ctx.env)
        if ctx.reasoning_effort:
            argv += ["--config", f"model_reasoning_effort={_toml_str(ctx.reasoning_effort)}"]
        argv += list(ctx.mcp_argv)
        if ctx.schema_path:
            argv += ["--output-schema", ctx.schema_path]
        argv += list(ctx.extra_args)
        return argv

    def new_parser(
        self,
        emit: Callable[[AgentEvent], None],
        *,
        expect_structured: bool,
    ) -> CodexStreamParser:
        """Build a stream parser for one run."""
        return CodexStreamParser(emit, expect_structured=expect_structured)

    def install_mcp(self, workspace: Path | None, servers: Sequence[McpServer]) -> McpInstallation:
        """Render *servers* as ``--config mcp_servers.*`` flags.

        Codex has no project-scoped config file: its loader reads MDM policy,
        system-managed config, ``~/.codex/config.toml``, and the session
        ``--config`` override layer. So the servers live in argv and nothing
        in the workspace is touched, which is why *workspace* is unused.
        """
        del workspace
        if not servers:
            return NoopInstallation()
        flags: list[str] = []
        for server in servers:
            flags += _server_flags(server)
        return FlagsInstallation(flags)

    def classify_exit(self, error: CliExitError, *, resumed: bool) -> AgentShimError:
        """Report a lost rollout as ``SessionResumeError``.

        Codex says so explicitly on stderr, so unlike Claude a resumed turn
        that failed for another reason keeps its generic exit error.
        """
        if not resumed or not _is_missing_rollout(error.stderr):
            return error
        session_id = _resumed_session_id(error.argv)
        if session_id is None:
            return error
        return SessionResumeError(
            error.argv, error.returncode, session_id, error.stdout, error.stderr
        )


def _is_missing_rollout(stderr: str) -> bool:
    return "thread/resume failed" in stderr and "no rollout found" in stderr


def _resumed_session_id(argv: Sequence[str]) -> str | None:
    args = list(argv)
    if "resume" not in args:
        return None
    index = args.index("resume")
    if index + 1 >= len(args):
        return None
    return args[index + 1]


def _shell_path_config(env: Mapping[str, str]) -> list[str]:
    """Preserve the launcher's PATH in the commands Codex spawns."""
    path = env.get("PATH")
    if not path:
        return []
    return ["--config", f"shell_environment_policy.set.PATH={_toml_str(path)}"]


def _server_flags(server: McpServer) -> list[str]:
    """Render one MCP server as dotted-path TOML overrides.

    TOML table keys are snake_case by convention, so ``vibesys-issues``
    becomes ``mcp_servers.vibesys_issues``.
    """
    key = server.name.replace("-", "_")
    prefix = f"mcp_servers.{key}"
    if isinstance(server, HttpMcpServer):
        if server.headers:
            raise ProviderCapabilityError(_HTTP_HEADERS_UNSUPPORTED)
        # A ``--config`` override carries the address and nothing else, so
        # Codex works the transport out from the endpoint itself.
        return ["--config", f"{prefix}.url={_toml_str(server.url)}"]
    flags = [
        "--config",
        f"{prefix}.command={_toml_str(server.command)}",
        "--config",
        f"{prefix}.args={_toml_array(list(server.args))}",
    ]
    for env_key, env_value in server.env.items():
        flags += ["--config", f"{prefix}.env.{env_key}={_toml_str(env_value)}"]
    return flags


def _toml_str(value: str) -> str:
    """Quote *value* as a TOML basic string literal."""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _toml_array(values: Sequence[str]) -> str:
    """Render a sequence of strings as a TOML inline array."""
    return "[" + ",".join(_toml_str(value) for value in values) + "]"


_CONFIG_OVERRIDE_RE = re.compile(r"^mcp_servers\.([^.=]+)\.(.+)$")
_TOML_STRING_RE = re.compile(r'"((?:[^"\\]|\\.)*)"')


def parse_mcp_servers(argv: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Recover the MCP servers rendered into *argv* by ``_server_flags``.

    The inverse of that renderer, kept next to it so the two cannot drift:
    walks the ``--config mcp_servers.<key>.<field>=<value>`` overrides this
    provider emits and rebuilds one canonical entry per server. A stdio
    server comes back as ``command``, ``args`` and ``env``; an HTTP one as
    ``url`` and ``transport`` (always ``"http"``, since a ``--config``
    override carries only the address and this provider cannot express any
    other transport).

    Args:
        argv: The argv a turn actually ran, as recorded on a ``CommandRequest``.

    Returns:
        One canonical entry per server, keyed by the dotted-path name
        ``_server_flags`` gave it (``-`` already mangled to ``_``).
    """
    servers: dict[str, dict[str, Any]] = {}
    for key, field, raw_value in _mcp_config_overrides(argv):
        _apply_mcp_override(servers.setdefault(key, {}), field, raw_value)
    for entry in servers.values():
        if "url" not in entry:
            entry.setdefault("args", [])
            entry.setdefault("env", {})
    return servers


def _mcp_config_overrides(argv: Sequence[str]) -> list[tuple[str, str, str]]:
    """Pull every ``mcp_servers.<key>.<field>=<value>`` override out of *argv*."""
    args = list(argv)
    overrides: list[tuple[str, str, str]] = []
    for index, arg in enumerate(args):
        if arg != "--config" or index + 1 >= len(args):
            continue
        path, sep, raw_value = args[index + 1].partition("=")
        if not sep:
            continue
        match = _CONFIG_OVERRIDE_RE.match(path)
        if match is not None:
            key, field = match.groups()
            overrides.append((key, field, raw_value))
    return overrides


def _apply_mcp_override(entry: dict[str, Any], field: str, raw_value: str) -> None:
    """Fold one dotted-path override into the entry being rebuilt for it."""
    if field == "command":
        entry["command"] = _parse_toml_str(raw_value)
    elif field == "args":
        entry["args"] = _parse_toml_array(raw_value)
    elif field == "url":
        entry["url"] = _parse_toml_str(raw_value)
        entry["transport"] = "http"
    elif field.startswith("env."):
        entry.setdefault("env", {})[field.removeprefix("env.")] = _parse_toml_str(raw_value)


def _parse_toml_str(literal: str) -> str:
    """Reverse ``_toml_str``: unescape one TOML basic string literal."""
    match = _TOML_STRING_RE.fullmatch(literal)
    if match is None:
        return literal
    return _unescape_toml(match.group(1))


def _parse_toml_array(literal: str) -> list[str]:
    """Reverse ``_toml_array``: unescape every string in a TOML inline array."""
    return [_unescape_toml(inner) for inner in _TOML_STRING_RE.findall(literal)]


def _unescape_toml(escaped: str) -> str:
    """Undo the backslash-escaping ``_toml_str`` applies to a string body."""
    result: list[str] = []
    index = 0
    while index < len(escaped):
        char = escaped[index]
        if char == "\\" and index + 1 < len(escaped):
            result.append(escaped[index + 1])
            index += 2
        else:
            result.append(char)
            index += 1
    return "".join(result)
