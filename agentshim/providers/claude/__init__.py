"""Claude Code."""

from __future__ import annotations

from .parser import ClaudeStreamParser
from .provider import PROFILE, ClaudeProvider, mcp_entry
from .sandbox import SandboxConfig, build_settings, resolve_sandbox
from .scripted import failure_lines, resume_failure_lines, scripted_lines
from .stream_transport import STREAM_PROFILE, ClaudeStreamTransport
from .user_hooks import ClaudeHook

__all__ = [
    "PROFILE",
    "STREAM_PROFILE",
    "ClaudeHook",
    "ClaudeProvider",
    "ClaudeStreamParser",
    "ClaudeStreamTransport",
    "SandboxConfig",
    "build_settings",
    "failure_lines",
    "mcp_entry",
    "resolve_sandbox",
    "resume_failure_lines",
    "scripted_lines",
]
