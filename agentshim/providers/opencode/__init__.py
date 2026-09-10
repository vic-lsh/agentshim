"""opencode."""

from __future__ import annotations

from .parser import OpencodeStreamParser
from .provider import PROFILE, OpencodeProvider, mcp_entry
from .scripted import resume_failure_lines, scripted_lines

__all__ = [
    "PROFILE",
    "OpencodeProvider",
    "OpencodeStreamParser",
    "mcp_entry",
    "resume_failure_lines",
    "scripted_lines",
]
