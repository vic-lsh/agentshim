"""Claude Code's native settings-file sandbox.

Claude Code sandboxes bash subprocesses at the OS level (bubblewrap on
Linux, Seatbelt on macOS) when its ``settings.json`` asks for it. This
builds that settings block; the provider passes it inline via ``--settings``.
The Claude process itself is not wrapped, so this is a provider option, not
a portable ``CliAgent`` argument.

See https://code.claude.com/docs/en/sandboxing.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ``absolute()`` rather than ``resolve()``: the hook path is only handed back to
# the interpreter, and an install reached through a symlinked tree should keep
# using the path it was imported from.
CONFINE_READS_HOOK = str(Path(__file__).absolute().parent / "hooks" / "confine_reads.py")

#: Claude Code otherwise cd's into a per-invocation scratch dir under
#: ``<project>/.local_tmp`` before every Bash call. That dir is outside the
#: sandbox's writable set, so every sandboxed Bash invocation fails with
#: EROFS before its command runs.
SANDBOX_ENV: dict[str, str] = {"CLAUDE_BASH_MAINTAIN_PROJECT_WORKING_DIR": "1"}

_CONFINED_TOOLS = "Read|Glob|Grep|Edit|Write|NotebookEdit"


def _strings() -> list[str]:
    return []


def _settings() -> dict[str, Any]:
    return {}


@dataclass
class SandboxConfig:
    """A subset of the ``sandbox`` key in Claude Code's ``settings.json``.

    Attributes:
        fail_if_unavailable: Exit with an error when the sandbox cannot start
            instead of silently running unsandboxed.
        auto_allow_bash: Auto-approve sandboxed bash commands; they are
            already constrained by the sandbox.
        allow_unsandboxed_commands: Whether the ``dangerouslyDisableSandbox``
            escape hatch stays available.
        excluded_commands: Commands that should run outside the sandbox.
        allow_write: Extra writable paths (cwd is always writable).
        deny_write: Paths sandboxed commands must not write.
        allow_read: Paths to re-allow inside a ``deny_read`` region.
        deny_read: Paths sandboxed commands must not read.
        allowed_domains: Outbound domains allowed for bash subprocesses. Does
            not affect Claude's own API calls.
        extra_settings: Raw keys merged into the ``sandbox`` block.
        confine_native_reads_to: Roots outside which Claude's own
            Read/Glob/Grep/Edit/Write tools are denied. The OS sandbox only
            wraps bash, so this is enforced by a ``PreToolUse`` hook. Empty
            means Claude's native tools may read anywhere.
    """

    fail_if_unavailable: bool = True
    auto_allow_bash: bool = True
    allow_unsandboxed_commands: bool = False
    excluded_commands: list[str] = field(default_factory=_strings)
    allow_write: list[str] = field(default_factory=_strings)
    deny_write: list[str] = field(default_factory=_strings)
    allow_read: list[str] = field(default_factory=_strings)
    deny_read: list[str] = field(default_factory=_strings)
    allowed_domains: list[str] = field(default_factory=_strings)
    extra_settings: dict[str, Any] = field(default_factory=_settings)
    confine_native_reads_to: list[str] = field(default_factory=_strings)


def resolve_sandbox(value: object) -> SandboxConfig | None:
    """Normalize the ``sandbox`` provider option to a config or ``None``.

    Takes ``object`` because the point is to reject what an untyped caller
    passed, not to restate the declared type.
    """
    if value is None or value is False:
        return None
    if value is True:
        return SandboxConfig()
    if isinstance(value, SandboxConfig):
        return value
    msg = f"sandbox must be bool or SandboxConfig, got {type(value).__name__}"
    raise TypeError(msg)


def build_settings(config: SandboxConfig) -> dict[str, Any]:
    """Build the ``settings.json`` payload that enables the sandbox."""
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
        settings["hooks"] = _confine_reads_hook(config.confine_native_reads_to)
    return settings


def _confine_reads_hook(roots: list[str]) -> dict[str, Any]:
    """Build the ``hooks`` block denying native-tool reads outside ``roots``.

    The hook runs through ``sys.executable`` so it uses the interpreter that
    ships agentshim, not whatever ``python3`` the agent's PATH resolves to
    and not the file's own executable bit.
    """
    resolved = [os.path.realpath(root) for root in roots]
    parts = [sys.executable, CONFINE_READS_HOOK, *resolved]
    command = " ".join(f'"{part}"' for part in parts)
    return {
        "PreToolUse": [
            {
                "matcher": _CONFINED_TOOLS,
                "hooks": [{"type": "command", "command": command}],
            }
        ]
    }
