"""Sandbox support for CLI coding agents.

Currently only ``ClaudeCodeCodingAgent`` supports sandboxing, via Claude
Code's native bubblewrap/Seatbelt sandbox (configured through its
``settings.json`` schema). Other providers raise ``NotImplementedError``
if sandboxing is requested.

See https://code.claude.com/docs/en/sandboxing for how the native sandbox
scopes bash subprocesses at the OS level (filesystem + network), without
wrapping the Claude process itself.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

_CONFINE_READS_HOOK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hooks", "confine_reads.py")


@dataclass
class SandboxConfig:
    """Claude Code sandbox configuration.

    Fields mirror a subset of the ``sandbox`` key in Claude Code's
    ``settings.json``. Only the knobs we actually need are exposed; pass
    additional keys through ``extra_settings`` if you need something not
    modeled here.

    Attributes:
        fail_if_unavailable: If True, Claude exits with an error when the
            sandbox can't start (missing ``bwrap``, unsupported platform)
            instead of silently falling back to unsandboxed execution.
            Recommended to keep True so sandbox behavior is a hard gate.
        auto_allow_bash: If True, sandboxed bash commands are auto-approved
            without permission prompts (they're already constrained by the
            sandbox).
        allow_unsandboxed_commands: If False, disables the
            ``dangerouslyDisableSandbox`` escape hatch entirely.
        excluded_commands: Commands that should run *outside* the sandbox
            (e.g. ``["docker *"]``).
        allow_write: Extra paths where sandboxed commands may write (cwd
            is always writable by default).
        deny_write: Paths sandboxed commands must not write to.
        allow_read: Paths to re-allow inside ``deny_read`` regions.
        deny_read: Paths sandboxed commands must not read.
        allowed_domains: Outbound network domains allowed for bash
            subprocesses (e.g. ``["github.com", "*.npmjs.org"]``).
            Does not affect Claude's own API calls.
        extra_settings: Raw dict merged into the ``sandbox`` settings
            block, for fields not covered above.
    """

    fail_if_unavailable: bool = True
    auto_allow_bash: bool = True
    allow_unsandboxed_commands: bool = False
    excluded_commands: list[str] = field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
    allow_write: list[str] = field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
    deny_write: list[str] = field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
    allow_read: list[str] = field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
    deny_read: list[str] = field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
    allowed_domains: list[str] = field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
    extra_settings: dict[str, Any] = field(default_factory=dict)  # pyright: ignore[reportUnknownVariableType]

    # The OS-level sandbox only wraps bash subprocesses, not Claude's built-in
    # Read/Glob/Grep/Edit/Write tools. When this list is non-empty we inject a
    # PreToolUse hook that denies any tool call whose target path is outside
    # the listed roots. Leave empty to allow Claude's native tools to read
    # anywhere (the default; matches unsandboxed behavior).
    confine_native_reads_to: list[str] = field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]


def resolve_sandbox(value: bool | SandboxConfig | None) -> SandboxConfig | None:
    """Normalize a user-supplied ``sandbox`` argument to a SandboxConfig or None."""
    if value is None or value is False:
        return None
    if value is True:
        return SandboxConfig()
    if isinstance(value, SandboxConfig):  # pyright: ignore[reportUnnecessaryIsInstance]
        return value
    raise TypeError(f"sandbox must be bool or SandboxConfig, got {type(value).__name__}")


def build_claude_sandbox_settings(config: SandboxConfig) -> dict[str, Any]:
    """Build the Claude Code ``settings.json`` payload that enables the sandbox."""
    sandbox: dict[str, Any] = {
        "enabled": True,
        "failIfUnavailable": config.fail_if_unavailable,
        "autoAllowBashIfSandboxed": config.auto_allow_bash,
        "allowUnsandboxedCommands": config.allow_unsandboxed_commands,
    }

    if config.excluded_commands:
        sandbox["excludedCommands"] = list(config.excluded_commands)

    filesystem: dict[str, Any] = {}
    if config.allow_write:
        filesystem["allowWrite"] = list(config.allow_write)
    if config.deny_write:
        filesystem["denyWrite"] = list(config.deny_write)
    if config.allow_read:
        filesystem["allowRead"] = list(config.allow_read)
    if config.deny_read:
        filesystem["denyRead"] = list(config.deny_read)
    if filesystem:
        sandbox["filesystem"] = filesystem

    if config.allowed_domains:
        sandbox["network"] = {"allowedDomains": list(config.allowed_domains)}

    sandbox.update(config.extra_settings)

    settings: dict[str, Any] = {"sandbox": sandbox}
    if config.confine_native_reads_to:
        settings["hooks"] = _build_confine_reads_hook(config.confine_native_reads_to)
    return settings


def _build_confine_reads_hook(roots: list[str]) -> dict[str, Any]:
    """Build the ``hooks`` block that denies native-tool reads outside ``roots``."""
    resolved = [os.path.realpath(r) for r in roots]
    # Quote each arg with double-quotes so paths with spaces survive shell parsing.
    args = " ".join(f'"{r}"' for r in resolved)
    command = f'"{_CONFINE_READS_HOOK}" {args}'
    return {
        "PreToolUse": [
            {
                "matcher": "Read|Glob|Grep|Edit|Write|NotebookEdit",
                "hooks": [{"type": "command", "command": command}],
            }
        ]
    }
