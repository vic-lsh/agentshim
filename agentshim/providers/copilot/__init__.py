"""GitHub Copilot CLI."""

from __future__ import annotations

from .parser import CopilotStreamParser
from .provider import PROFILE, CopilotProvider, mcp_entry, parse_mcp_servers
from .scripted import resume_failure_lines, scripted_lines

__all__ = [
    "PROFILE",
    "CopilotProvider",
    "CopilotStreamParser",
    "mcp_entry",
    "parse_mcp_servers",
    "resume_failure_lines",
    "scripted_lines",
]
