#!/usr/bin/env python3
"""Claude Code ``PreToolUse`` hook that denies file reads/edits outside an allowlist.

Wired up via ``SandboxConfig.confine_native_reads_to`` — see
``agentshim/sandbox.py``. The hook is invoked by Claude Code for each
tool call matching ``Read|Glob|Grep|Edit|Write``. It parses the JSON input
on stdin, resolves the tool's target path, and rejects the call if the path
doesn't land under one of the allowed roots passed on argv.

Usage::

    confine_reads.py <allowed_root> [<allowed_root> ...]
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any


def _candidate_path(tool_input: dict[str, Any]) -> str | None:
    """Best-effort extraction of the filesystem target from a tool_input blob."""
    for key in ("file_path", "path", "notebook_path"):
        value = tool_input.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _is_under(path: str, roots: list[str]) -> bool:
    for root in roots:
        if path == root:
            return True
        if path.startswith(root.rstrip(os.sep) + os.sep):
            return True
    return False


def main() -> int:
    if len(sys.argv) < 2:
        # Misconfigured hook — don't block the tool, just let it through.
        return 0

    roots = [os.path.realpath(p) for p in sys.argv[1:]]

    try:
        payload: dict[str, Any] = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0

    tool_input: dict[str, Any] = payload.get("tool_input") or {}
    candidate = _candidate_path(tool_input)
    if not candidate:
        # Tool call has no path argument (e.g. Glob without `path` defaults to cwd).
        return 0

    target = os.path.realpath(os.path.abspath(candidate))
    if _is_under(target, roots):
        return 0

    decision = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": (
                f"Path {target!r} is outside the sandbox roots {roots}. Stay inside the working directory."
            ),
        }
    }
    json.dump(decision, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
