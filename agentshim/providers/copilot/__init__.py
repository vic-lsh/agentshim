"""GitHub Copilot CLI."""

from __future__ import annotations

from agentshim.providers.copilot.parser import CopilotStreamParser, fold_usage
from agentshim.providers.copilot.provider import PROFILE, CopilotProvider
from agentshim.providers.copilot.scripted import scripted_lines

__all__ = [
    "PROFILE",
    "CopilotProvider",
    "CopilotStreamParser",
    "fold_usage",
    "scripted_lines",
]
