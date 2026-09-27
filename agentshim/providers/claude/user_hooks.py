"""Caller-supplied Claude Code hooks.

Claude Code runs a hook command on an event such as ``PreToolUse`` and acts
on the JSON it prints, which is how a caller enforces a policy the sandbox
cannot express, for example refusing one executable in Bash. agentshim
already passes its own settings inline through ``--settings``; a second
``--settings`` in ``extra_args`` would compete with that one, so the provider
takes hooks as an option and merges them into the single settings object.

See https://code.claude.com/docs/en/hooks.
"""

from __future__ import annotations

import math
import re
import shlex
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

#: Claude Code's hook events are PascalCase names. New events appear between
#: CLI releases, so the shape is checked rather than a closed list.
_EVENT_RE = re.compile(r"[A-Z][A-Za-z]*")


@dataclass(frozen=True)
class ClaudeHook:
    """One command Claude Code runs on a hook event.

    Attributes:
        event: The hook event, such as ``PreToolUse`` or ``PostToolUse``.
        command: The hook's argv. Claude Code runs hooks through a shell, so
            agentshim quotes it with ``shlex.join``: every element reaches the
            process literally, whatever it contains. Use an absolute
            executable path; the agent's PATH is not the caller's. Any
            sequence of strings is accepted and stored as a tuple.
        matcher: For tool events, the tool-name pattern the hook applies to,
            such as ``Bash`` or ``Edit|Write``. ``None`` matches every tool.
        timeout_s: Seconds Claude Code allows the hook before giving up;
            ``None`` keeps the CLI default.
    """

    event: str
    command: Sequence[str]
    matcher: str | None = None
    timeout_s: float | None = None

    def __post_init__(self) -> None:
        """Reject a hook Claude Code would refuse or run incorrectly."""
        _check_event(self.event)
        object.__setattr__(self, "command", _as_argv(self.command))
        _check_matcher(self.matcher)
        if self.timeout_s is not None and not _positive_finite(self.timeout_s):
            msg = f"timeout_s must be a positive finite number, got {self.timeout_s!r}"
            raise ValueError(msg)


def _check_event(event: object) -> None:
    if not isinstance(event, str) or not _EVENT_RE.fullmatch(event):
        msg = f"event must be a PascalCase hook event name, got {event!r}"
        raise ValueError(msg)


def _check_matcher(matcher: object) -> None:
    if matcher is not None and (not isinstance(matcher, str) or not matcher):
        msg = f"matcher must be a non-empty string or None, got {matcher!r}"
        raise ValueError(msg)


def _as_argv(value: object) -> tuple[str, ...]:
    """Validate a hook command and freeze it into a tuple.

    A bare string is rejected rather than split: ``"python3 hook.py"`` would
    otherwise become one argv element naming a file that does not exist.
    """
    if isinstance(value, str) or not isinstance(value, Sequence):
        msg = f"command must be an argv sequence, not {type(value).__name__}"
        raise TypeError(msg)
    argv = tuple(cast("Sequence[object]", value))
    if not argv:
        msg = "command must not be empty"
        raise ValueError(msg)
    for arg in argv:
        if not isinstance(arg, str):
            msg = f"command elements must be str, got {type(arg).__name__}"
            raise TypeError(msg)
        if "\x00" in arg:
            msg = f"command element contains a NUL byte: {arg!r}"
            raise ValueError(msg)
        try:
            arg.encode("utf-8")
        except UnicodeEncodeError:
            msg = f"command element is not valid UTF-8 text: {arg!r}"
            raise ValueError(msg) from None
    return cast("tuple[str, ...]", argv)


def _positive_finite(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value) and value > 0


def resolve_hooks(value: object) -> tuple[ClaudeHook, ...]:
    """Normalize the ``hooks`` provider option to a tuple of hooks."""
    if isinstance(value, ClaudeHook) or not isinstance(value, Sequence):
        msg = f"hooks must be a sequence of ClaudeHook, got {type(value).__name__}"
        raise TypeError(msg)
    hooks = tuple(cast("Sequence[object]", value))
    for hook in hooks:
        if not isinstance(hook, ClaudeHook):
            msg = f"hooks must contain ClaudeHook, got {type(hook).__name__}"
            raise TypeError(msg)
    return cast("tuple[ClaudeHook, ...]", hooks)


def render_hooks(hooks: Sequence[ClaudeHook]) -> dict[str, list[dict[str, Any]]]:
    """Render *hooks* as a settings ``hooks`` block, one entry per hook.

    Entries keep the caller's order within each event. One entry per hook,
    rather than grouping by matcher, keeps every hook's timeout its own.
    """
    block: dict[str, list[dict[str, Any]]] = {}
    for hook in hooks:
        handler: dict[str, Any] = {"type": "command", "command": shlex.join(hook.command)}
        if hook.timeout_s is not None:
            handler["timeout"] = hook.timeout_s
        entry: dict[str, Any] = {"hooks": [handler]}
        if hook.matcher is not None:
            entry["matcher"] = hook.matcher
        block.setdefault(hook.event, []).append(entry)
    return block


def merge_hooks(*blocks: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    """Concatenate hook blocks event by event, earlier blocks first."""
    merged: dict[str, list[dict[str, Any]]] = {}
    for block in blocks:
        for event, entries in block.items():
            merged.setdefault(event, []).extend(entries)
    return merged
