"""opencode."""

from __future__ import annotations

from .parser import OpencodeStreamParser, fold_usage
from .provider import PROFILE, OpencodeProvider, mcp_entry
from .scripted import scripted_lines

__all__ = [
    "PROFILE",
    "OpencodeProvider",
    "OpencodeStreamParser",
    "fold_usage",
    "mcp_entry",
    "scripted_lines",
]
