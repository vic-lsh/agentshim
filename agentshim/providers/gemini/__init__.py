"""Gemini CLI."""

from __future__ import annotations

from .parser import GeminiStreamParser, fold_usage
from .provider import PROFILE, GeminiProvider, mcp_entry
from .scripted import scripted_lines

__all__ = [
    "PROFILE",
    "GeminiProvider",
    "GeminiStreamParser",
    "fold_usage",
    "mcp_entry",
    "scripted_lines",
]
