"""opencode."""

from __future__ import annotations

from .parser import OpencodeStreamParser, fold_tokens
from .provider import PROFILE, OpencodeProvider, mcp_entry
from .scripted import scripted_lines

__all__ = [
    "PROFILE",
    "OpencodeProvider",
    "OpencodeStreamParser",
    "fold_tokens",
    "mcp_entry",
    "scripted_lines",
]
