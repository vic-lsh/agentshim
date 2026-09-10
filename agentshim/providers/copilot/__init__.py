"""GitHub Copilot CLI."""

from __future__ import annotations

from .parser import CopilotStreamParser, fold_usage
from .provider import PROFILE, CopilotProvider, mcp_entry
from .scripted import scripted_lines

__all__ = [
    "PROFILE",
    "CopilotProvider",
    "CopilotStreamParser",
    "fold_usage",
    "mcp_entry",
    "scripted_lines",
]
