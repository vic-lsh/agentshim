"""The Codex provider: argv, parser, MCP install, exit classification."""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING, Any

from agentshim.core.errors import ProviderCapabilityError, SessionResumeError
from agentshim.core.home import state_root
from agentshim.core.mcp import FlagsInstallation, HttpMcpServer, NoopInstallation
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

from ._toml import toml_array, toml_str, unescape_toml
from .mcp_scope import session_scope_overrides
from .parser import CodexStreamParser
from .rules import RULES_FILENAME
from .sandbox import CodexSandboxConfig, resolve_sandbox, sandbox_overrides
from .skills import project_scope_overrides

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
    # ``exec --json`` lists no skills; a load is inferred from a shell read
    # of a ``SKILL.md`` (``skills.py``).
    skill_invocation=SkillSignal.INFERRED,
    skill_scopes=frozenset({SkillScope.ALL, SkillScope.PROJECT}),
    mcp_scopes=frozenset({McpScope.ALL, McpScope.SESSION}),
    config_scopes=frozenset({ConfigScope.ALL, ConfigScope.PROJECT}),
    # Codex reads AGENTS.md, hooks.json and memories/ from $CODEX_HOME with no
    # flag to skip them, so ConfigScope.PROJECT needs a home of its own.
    config_home_files=("auth.json",),
    container_install=(_NODE_INSTALL, _CODEX_INSTALL),
    # Documented Codex CLI variable that relocates ~/.codex (``codex --help``:
    # "Layer $CODEX_HOME/<name>.config.toml on top of the base user config";
    # "auth still uses CODEX_HOME").
    state_root_env="CODEX_HOME",
    # Authentication is self-contained in auth.json.  config.toml is user
    # policy and preferences, so consumers that copy or mount auth state must
    # not needlessly expose it to an agent process.
    auth_files=(".codex/auth.json",),
    # CLI_FLAGS: Codex takes MCP servers as --config overrides, never a file.
    mcp_config_file=None,
)


#: Turns off both Codex's sandbox and its approval prompts.
BYPASS_FLAG = "--dangerously-bypass-approvals-and-sandbox"

#: Keeps user and project ``.rules`` files out of a sandboxed turn.
IGNORE_RULES_FLAG = "--ignore-rules"

#: ``ConfigScope.PROJECT``: ``$CODEX_HOME/config.toml`` (profiles, notify,
#: ``developer_instructions``, project trust) is not loaded, and the memories
#: Codex would otherwise write and reread across sessions in the home stay
#: off. The global ``AGENTS.md``, ``hooks.json`` and existing memories are
#: kept out by the dedicated home itself (``prepare_config_home``).
PROJECT_CONFIG_FLAGS = ("--ignore-user-config", "--disable", "memories")


class CodexProvider:
    """Codex (``codex exec --json``). ``sandbox`` is a provider option."""

    profile = PROFILE

    def __init__(self, *, sandbox: CodexSandboxConfig | None = None) -> None:
        """Fix the sandbox for every turn this provider runs.

        ``None``, the default, bypasses Codex's sandbox and approvals, for a
        caller that isolates the whole process itself. A
        ``CodexSandboxConfig`` keeps the CLI's own sandbox on instead.
        """
        self.sandbox: CodexSandboxConfig | None = resolve_sandbox(sandbox)

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
        argv += self._sandbox_argv(ctx)
        argv += ["--skip-git-repo-check", "--json"]
        if ctx.model:
            argv += ["--model", ctx.model]
        argv += _shell_path_config(ctx.env)
        if ctx.config_scope is ConfigScope.PROJECT:
            _check_config_home(ctx)
            argv += PROJECT_CONFIG_FLAGS
        argv += _scope_overrides(ctx)
        if ctx.reasoning_effort:
            argv += ["--config", f"model_reasoning_effort={toml_str(ctx.reasoning_effort)}"]
        argv += list(ctx.mcp_argv)
        if ctx.schema_path:
            argv += ["--output-schema", ctx.schema_path]
        argv += list(ctx.extra_args)
        return argv

    def _sandbox_argv(self, ctx: ArgvContext) -> list[str]:
        """Render the sandbox, and decide which exec-policy rules may apply.

        Codex runs a command that an ``allow`` rule matches outside its
        sandbox, and loads rules from ``$CODEX_HOME/rules`` and trusted
        projects. Without exemptions ``--ignore-rules`` keeps every such file
        out, the user's own included. With them the rules have to load, so
        the home they come from is checked instead.
        """
        if self.sandbox is None:
            return [BYPASS_FLAG]
        argv: list[str] = []
        for key, value in sandbox_overrides(self.sandbox):
            argv += ["--config", f"{key}={value}"]
        if self.sandbox.excluded_commands:
            _check_rules_home(self.sandbox, ctx)
        else:
            argv.append(IGNORE_RULES_FLAG)
        return argv

    def new_parser(
        self,
        emit: Callable[[AgentEvent], None],
        *,
        expect_structured: bool,
    ) -> CodexStreamParser:
        """Build a stream parser for one run."""
        return CodexStreamParser(
            emit,
            expect_structured=expect_structured,
        )

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
            error.argv,
            error.returncode,
            session_id,
            error.stdout,
            error.stderr,
            detail=error.detail,
        )


def _check_rules_home(config: CodexSandboxConfig, ctx: ArgvContext) -> None:
    """Refuse a ``CODEX_HOME`` that cannot hold the exemptions safely.

    The rules are read from ``$CODEX_HOME/rules`` on every turn, so a home the
    sandbox lets commands write would let the model install a rule exempting
    anything, and have it apply from the next turn on.
    """
    home = ctx.env.get("CODEX_HOME")
    if not home or not os.path.isabs(home):  # noqa: PTH117 - a str contract, not a Path
        msg = (
            "excluded_commands are read from $CODEX_HOME/rules/"
            f"{RULES_FILENAME}: run the turn with CODEX_HOME set to the absolute "
            f"path of a dedicated home prepared with install_rules (got {home!r})"
        )
        raise ProviderCapabilityError(msg)
    for directory in _writable_dirs(config, ctx):
        if _is_within(home, directory) or _is_within(directory, home):
            msg = (
                f"CODEX_HOME {home} overlaps {directory}, which the sandbox lets "
                "commands write, so a command could add rules exempting itself"
            )
            raise ProviderCapabilityError(msg)


def _writable_dirs(config: CodexSandboxConfig, ctx: ArgvContext) -> list[str]:
    """The directories *config* lets sandboxed commands write, where known."""
    if config.mode != "workspace-write":
        return []
    dirs = list(config.writable_roots)
    if ctx.cwd:
        dirs.append(ctx.cwd)
    if config.writable_tmp:
        dirs.append("/tmp")  # noqa: S108 - Codex's own writable /tmp
        tmpdir = ctx.env.get("TMPDIR")
        if tmpdir:
            dirs.append(tmpdir)
    return dirs


def _check_config_home(ctx: ArgvContext) -> None:
    """Refuse ``ConfigScope.PROJECT`` unless the turn runs in a home of its own.

    The user's global ``AGENTS.md``, hooks and memories live in the state root
    and no flag skips them, so a turn still pointed at the user's root would
    load them while claiming to be isolated.
    """
    home = ctx.env.get("CODEX_HOME")
    if not home or not os.path.isabs(home):  # noqa: PTH117 - a str contract, not a Path
        msg = (
            "ConfigScope.PROJECT needs CODEX_HOME set to the absolute path of a "
            f"dedicated home prepared with prepare_config_home (got {home!r})"
        )
        raise ProviderCapabilityError(msg)
    user_root = state_root(PROFILE, {key: v for key, v in ctx.env.items() if key != "CODEX_HOME"})
    if user_root is not None and os.path.realpath(home) == os.path.realpath(user_root):
        msg = f"ConfigScope.PROJECT needs a dedicated CODEX_HOME, not the user's own {home}"
        raise ProviderCapabilityError(msg)


def _is_within(path: str, directory: str) -> bool:
    resolved, parent = os.path.realpath(path), os.path.realpath(directory)
    return resolved == parent or resolved.startswith(parent.rstrip(os.sep) + os.sep)


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
    return ["--config", f"shell_environment_policy.set.PATH={toml_str(path)}"]


def _scope_overrides(ctx: ArgvContext) -> list[str]:
    """The ``--config`` flags the session's skill and MCP scopes call for.

    Both scopes switch ``features.plugins`` off; the flag is emitted once.
    """
    flags: list[str] = []
    if ctx.skill_scope is SkillScope.PROJECT:
        flags += project_scope_overrides(ctx.env)
    if ctx.mcp_scope is McpScope.SESSION:
        given = {server.name.replace("-", "_") for server in ctx.mcp_servers}
        flags += session_scope_overrides(ctx.env, ctx.cwd, given)
    pairs = [flags[i : i + 2] for i in range(0, len(flags), 2)]
    return [item for pair in dict.fromkeys(map(tuple, pairs)) for item in pair]


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
        flags = [
            "--config",
            f"{prefix}.url={toml_str(server.url)}",
            "--config",
            f"{prefix}.required=true",
        ]
        return flags + _timeout_flags(prefix, server)
    flags = [
        "--config",
        f"{prefix}.command={toml_str(server.command)}",
        "--config",
        f"{prefix}.args={toml_array(list(server.args))}",
        "--config",
        f"{prefix}.required=true",
    ]
    for env_key, env_value in server.env.items():
        flags += ["--config", f"{prefix}.env.{env_key}={toml_str(env_value)}"]
    return flags + _timeout_flags(prefix, server)


def _timeout_flags(prefix: str, server: McpServer) -> list[str]:
    """Render Codex's per-server timeouts as invocation overrides."""
    flags: list[str] = []
    if server.startup_timeout_s is not None:
        flags += ["--config", f"{prefix}.startup_timeout_sec={server.startup_timeout_s}"]
    if server.tool_timeout_s is not None:
        flags += ["--config", f"{prefix}.tool_timeout_sec={server.tool_timeout_s}"]
    return flags


_CONFIG_OVERRIDE_RE = re.compile(r"^mcp_servers\.([^.=]+)\.(.+)$")
_TOML_STRING_RE = re.compile(r'"((?:[^"\\]|\\.)*)"')


_DISABLED = "\0disabled"
"""Scratch key marking an entry an ``enabled=false`` override switched off."""


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
    # ``McpScope.SESSION`` also emits overrides that disable the servers the
    # user configured (their transport plus ``enabled=false``); those are not
    # servers the turn was given, so they are not reported.
    servers = {
        key: entry
        for key, entry in servers.items()
        if not entry.pop(_DISABLED, False) and ("command" in entry or "url" in entry)
    }
    for entry in servers.values():
        if "url" not in entry:
            entry.setdefault("args", [])
            entry.setdefault("env", {})
    return servers


def _mcp_config_overrides(argv: Sequence[str]) -> list[tuple[str, str, str]]:
    """Pull every ``mcp_servers.<key>.<field>=<value>`` override out of *argv*."""
    overrides: list[tuple[str, str, str]] = []
    for path, raw_value in _config_overrides(argv):
        match = _CONFIG_OVERRIDE_RE.match(path)
        if match is not None:
            key, field = match.groups()
            overrides.append((key, field, raw_value))
    return overrides


def _apply_mcp_override(entry: dict[str, Any], field: str, raw_value: str) -> None:
    """Fold one dotted-path override into the entry being rebuilt for it."""
    if field == "command":
        entry["command"] = _parsetoml_str(raw_value)
    elif field == "args":
        entry["args"] = _parsetoml_array(raw_value)
    elif field == "url":
        entry["url"] = _parsetoml_str(raw_value)
        entry["transport"] = "http"
    elif field == "enabled":
        entry[_DISABLED] = raw_value.strip() == "false"
    elif field == "tool_timeout_sec":
        entry["tool_timeout_s"] = float(raw_value)
    elif field == "startup_timeout_sec":
        entry["startup_timeout_s"] = float(raw_value)
    elif field.startswith("env."):
        entry.setdefault("env", {})[field.removeprefix("env.")] = _parsetoml_str(raw_value)


def _parsetoml_str(literal: str) -> str:
    """Reverse ``toml_str``: unescape one TOML basic string literal."""
    match = _TOML_STRING_RE.fullmatch(literal)
    if match is None:
        return literal
    return unescape_toml(match.group(1))


def _parsetoml_array(literal: str) -> list[str]:
    """Reverse ``toml_array``: unescape every string in a TOML inline array."""
    return [unescape_toml(inner) for inner in _TOML_STRING_RE.findall(literal)]


_SANDBOX_KEYS = {
    "sandbox_mode",
    "sandbox_workspace_write.writable_roots",
    "sandbox_workspace_write.network_access",
    "sandbox_workspace_write.exclude_slash_tmp",
}


def parse_sandbox(argv: Sequence[str]) -> CodexSandboxConfig | None:
    """Recover the sandbox rendered into *argv* by ``CodexProvider``.

    The inverse of ``CodexProvider._sandbox_argv``, kept next to it so the
    two cannot drift.

    Args:
        argv: The argv a turn actually ran, as recorded on a ``CommandRequest``.

    Returns:
        ``None`` when the turn bypassed Codex's sandbox, else the config it
        imposed. ``excluded_commands`` is always empty: exemptions live in the
        rules file under ``CODEX_HOME``, not in argv (read them back with
        ``parse_rules``). The absence of ``--ignore-rules`` is the argv's only
        sign that the turn loaded rules.

    Raises:
        ValueError: *argv* neither bypasses the sandbox nor selects a mode.
    """
    if BYPASS_FLAG in argv:
        return None
    overrides = {key: value for key, value in _config_overrides(argv) if key in _SANDBOX_KEYS}
    if "sandbox_mode" not in overrides:
        msg = f"argv neither bypasses nor selects a Codex sandbox: {list(argv)!r}"
        raise ValueError(msg)
    prefix = "sandbox_workspace_write"
    return CodexSandboxConfig(
        mode=_parsetoml_str(overrides["sandbox_mode"]),  # pyright: ignore[reportArgumentType]
        writable_roots=_parsetoml_array(overrides.get(f"{prefix}.writable_roots", "[]")),
        network_access=overrides.get(f"{prefix}.network_access") == "true",
        writable_tmp=overrides.get(f"{prefix}.exclude_slash_tmp", "false") == "false",
    )


def _config_overrides(argv: Sequence[str]) -> list[tuple[str, str]]:
    """Pull every ``--config key=value`` pair out of *argv*, in order."""
    args = list(argv)
    pairs: list[tuple[str, str]] = []
    for index, arg in enumerate(args[:-1]):
        if arg == "--config":
            key, sep, value = args[index + 1].partition("=")
            if sep:
                pairs.append((key, value))
    return pairs
