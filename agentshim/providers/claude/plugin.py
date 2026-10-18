"""Claude's complete catalog declaration."""

from __future__ import annotations

from agentshim.core.plugin import ProviderPlugin, ProviderTestSupport

from .model_data import MODEL_ENTRIES, PRICING_ENTRIES
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
    model_vendor="anthropic",
    pricing_vendor="anthropic",
    model_entries=MODEL_ENTRIES,
    pricing_entries=PRICING_ENTRIES,
    stream_factory=ClaudeStreamTransport,
    testing=ProviderTestSupport(
        scripted_lines=scripted_lines,
        resume_failure_lines=resume_failure_lines,
        failure_lines=failure_lines,
        installed_mcp_servers=installed_mcp_servers,
        probe_fixture=probe_fixture,
    ),
)
