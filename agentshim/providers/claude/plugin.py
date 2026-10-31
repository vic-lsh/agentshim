"""Claude's complete catalog declaration."""

from __future__ import annotations

from agentshim.core.plugin import ProviderTestSupport
from agentshim.providers.static_plugin import StaticProviderPlugin

from .model_data import MODEL_ENTRIES, PRICING_ENTRIES, RATES, UNPRICEABLE_MODELS
from .probe import PROBE
from .provider import PROFILE, ClaudeProvider
from .scripted import failure_lines, resume_failure_lines, scripted_lines
from .stream_transport import ClaudeStreamTransport
from .testing import installed_mcp_servers, probe_fixture
from .transcripts import TRANSCRIPTS

PLUGIN = StaticProviderPlugin(
    name="claude",
    profile=PROFILE,
    provider_factory=ClaudeProvider,
    readiness=PROBE,
    model_vendor="anthropic",
    pricing_vendor="anthropic",
    model_entries=MODEL_ENTRIES,
    pricing_entries=PRICING_ENTRIES,
    rates=RATES,
    unpriceable_models=UNPRICEABLE_MODELS,
    transcript_format=TRANSCRIPTS,
    stream_factory=ClaudeStreamTransport,
    testing=ProviderTestSupport(
        scripted_lines=scripted_lines,
        resume_failure_lines=resume_failure_lines,
        failure_lines=failure_lines,
        installed_mcp_servers=installed_mcp_servers,
        probe_fixture=probe_fixture,
    ),
)
