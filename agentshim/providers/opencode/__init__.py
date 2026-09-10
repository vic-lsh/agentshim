"""opencode."""

from __future__ import annotations

from .parser import OpencodeStreamParser
from .provider import PROFILE, OpencodeProvider, mcp_entry
from .scripted import scripted_lines

__all__ = [
    "PROFILE",
    "OpencodeProvider",
    "OpencodeStreamParser",
    "mcp_entry",
    "scripted_lines",
]
