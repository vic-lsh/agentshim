"""GitHub Copilot CLI."""

from __future__ import annotations

from .parser import CopilotStreamParser
from .provider import PROFILE, CopilotProvider, mcp_entry
from .scripted import scripted_lines

__all__ = [
    "PROFILE",
    "CopilotProvider",
    "CopilotStreamParser",
    "mcp_entry",
    "scripted_lines",
]
