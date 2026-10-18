"""Copilot's complete catalog declaration."""

from __future__ import annotations

from agentshim.core.plugin import ProviderTestSupport
from agentshim.providers.static_plugin import StaticProviderPlugin

from .probe import PROBE
from .provider import PROFILE, CopilotProvider
from .scripted import resume_failure_lines, scripted_lines
from .testing import installed_mcp_servers, probe_fixture

PLUGIN = StaticProviderPlugin(
    name="copilot",
    profile=PROFILE,
    provider_factory=CopilotProvider,
    readiness=PROBE,
    testing=ProviderTestSupport(
        scripted_lines=scripted_lines,
        resume_failure_lines=resume_failure_lines,
        installed_mcp_servers=installed_mcp_servers,
        probe_fixture=probe_fixture,
    ),
)
