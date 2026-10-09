"""The command line and environment of one long-lived ``claude`` process.

Everything here is a pure function of the conversation's spec and of the few
settings that can only change by restarting the process (``ProcessConfig``),
so the transport can compare two configs to know whether a turn needs a new
process.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from agentshim.core.permissions import NativeMode
from agentshim.core.profile import ConfigScope, McpScope, SkillScope

from .provider import (
    MCP_SERVER_KEY,
    NO_AUTO_MEMORY_SETTINGS,
    PROJECT_SETTING_SOURCES,
    STRICT_MCP_FLAG,
    mcp_entry,
)
from .sandbox import SANDBOX_ENV, build_settings, workspace_write_settings

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from agentshim.core.conversation import ConversationSpec
    from agentshim.core.mcp import McpServer

    from .user_hooks import ClaudeHook


@dataclass(frozen=True)
class ProcessConfig:
    """The per-turn choices that are fixed for the life of a process.

    Claude Code takes these as flags or environment at start, so a turn that
    wants different ones needs a new process (resuming the conversation).
    """

    #: The output schema as compact JSON, or ``None`` for prose answers.
    schema: str | None = None
    reasoning_effort: str | None = None
    mcp_servers: tuple[McpServer, ...] = ()
    extra_args: tuple[str, ...] = ()
    #: Extra environment for the process, as sorted ``(name, value)`` pairs.
    env: tuple[tuple[str, str], ...] = ()


def stream_argv(
    binary: str,
    spec: ConversationSpec,
    config: ProcessConfig,
    *,
    resume_id: str | None,
    hooks: Sequence[ClaudeHook] = (),
) -> list[str]:
    """Build the argv of a process speaking ``stream-json`` both ways.

    There is no ``-p`` and no prompt: stream-json input implies print mode and
    prompts arrive as user messages on stdin. Claude Code's own system prompt
    is kept (no ``--system-prompt`` flag). ``--permission-prompts none`` makes
    the CLI deny anything that would ask; the transport still answers any
    permission request that arrives anyway.
    """
    argv = [
        binary,
        "--output-format",
        "stream-json",
        "--verbose",
        "--input-format",
        "stream-json",
        "--permission-prompts",
        "none",
        # Echoes each user message when the CLI takes it into a turn: the signal
        # that a steered message was consumed (see ``stream_transport``).
        "--replay-user-messages",
    ]
    if spec.model:
        argv += ["--model", spec.model]
    if resume_id:
        # The equals form: a value that starts with a dash must not be parsed
        # as another flag.
        argv.append(f"--resume={resume_id}")
    argv += _permission_argv(spec)
    if spec.skill_scope is SkillScope.PROJECT or spec.config_scope is ConfigScope.PROJECT:
        argv += PROJECT_SETTING_SOURCES
    if config.reasoning_effort:
        argv += ["--effort", config.reasoning_effort]
    if config.schema:
        argv += ["--json-schema", config.schema]
    settings = _settings(spec, hooks)
    if settings:
        argv += ["--settings", json.dumps(settings)]
    argv += _mcp_argv(spec.mcp_scope, config.mcp_servers)
    argv += config.extra_args
    return argv


def stream_env(
    base: Mapping[str, str], spec: ConversationSpec, config: ProcessConfig
) -> dict[str, str]:
    """The environment of the process: *base* minus the nested-session marker, plus needs.

    ``CLAUDECODE`` is dropped because a ``claude`` that sees it believes it
    runs inside another Claude Code session and changes behaviour. A sandboxed
    mode adds what Claude's bash sandbox needs to start.
    """
    env = {name: value for name, value in base.items() if name != "CLAUDECODE"}
    if spec.permissions.mode is NativeMode.WORKSPACE_WRITE:
        env.update(SANDBOX_ENV)
    env.update(config.env)
    return env


def _permission_argv(spec: ConversationSpec) -> list[str]:
    permissions = spec.permissions
    if permissions.mode is NativeMode.BYPASS:
        return ["--permission-mode", "bypassPermissions"]
    # Workspace write: file edits inside the working directory (and the extra
    # roots, which ``--add-dir`` adds to it) are accepted without a prompt.
    # Everything else that would prompt is denied by ``--permission-prompts none``.
    argv = ["--permission-mode", "acceptEdits"]
    for root in permissions.writable_roots:
        argv += ["--add-dir", root]
    return argv


def _settings(spec: ConversationSpec, hooks: Sequence[ClaudeHook]) -> dict[str, object]:
    permissions = spec.permissions
    if permissions.mode is NativeMode.WORKSPACE_WRITE:
        settings: dict[str, object] = dict(
            workspace_write_settings(spec.cwd, permissions.writable_roots, hooks=hooks)
        )
    else:
        settings = dict(build_settings(None, hooks=hooks))
    if spec.config_scope is ConfigScope.PROJECT:
        settings.update(NO_AUTO_MEMORY_SETTINGS)
    return settings


def _mcp_argv(scope: McpScope, servers: Sequence[McpServer]) -> list[str]:
    argv = [STRICT_MCP_FLAG] if scope is McpScope.SESSION else []
    if servers:
        config = {MCP_SERVER_KEY: {server.name: mcp_entry(server) for server in servers}}
        argv += ["--mcp-config", json.dumps(config)]
    return argv
