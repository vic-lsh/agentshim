"""Claude's complete catalog declaration."""

from __future__ import annotations

from agentshim.providers.catalog import ProviderPlugin, ProviderTestSupport

from .probe import PROBE
from .provider import PROFILE, ClaudeProvider
from .scripted import failure_lines, resume_failure_lines, scripted_lines
from .stream_transport import ClaudeStreamTransport
from .testing import installed_mcp_servers, probe_fixture

PLUGIN = ProviderPlugin(
    name="claude",
    profile=PROFILE,
    provider_factory=ClaudeProvider,
    probe=PROBE,
    stream_factory=ClaudeStreamTransport,
    testing=ProviderTestSupport(
        scripted_lines=scripted_lines,
        resume_failure_lines=resume_failure_lines,
        failure_lines=failure_lines,
        installed_mcp_servers=installed_mcp_servers,
        probe_fixture=probe_fixture,
    ),
)
