"""Copilot's complete catalog declaration."""

from __future__ import annotations

from agentshim.providers.catalog import ProviderPlugin, ProviderTestSupport

from .probe import PROBE
from .provider import PROFILE, CopilotProvider
from .scripted import resume_failure_lines, scripted_lines
from .testing import installed_mcp_servers, probe_fixture

PLUGIN = ProviderPlugin(
    name="copilot",
    profile=PROFILE,
    provider_factory=CopilotProvider,
    probe=PROBE,
    testing=ProviderTestSupport(
        scripted_lines=scripted_lines,
        resume_failure_lines=resume_failure_lines,
        installed_mcp_servers=installed_mcp_servers,
        probe_fixture=probe_fixture,
    ),
)
