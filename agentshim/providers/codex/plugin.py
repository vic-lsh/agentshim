"""Codex's complete catalog declaration."""

from __future__ import annotations

from agentshim.core.plugin import ProviderTestSupport
from agentshim.providers.static_plugin import StaticProviderPlugin

from .app_server import CodexAppServerTransport
from .limits import CodexLimitsReader
from .model_data import MODEL_ENTRIES, PRICING_ENTRIES, RATES
from .probe import PROBE
from .provider import PROFILE, CodexProvider
from .scripted import failure_lines, resume_failure_lines, scripted_lines
from .testing import installed_mcp_servers, probe_fixture
from .transcripts import TRANSCRIPTS

PLUGIN = StaticProviderPlugin(
    name="codex",
    profile=PROFILE,
    provider_factory=CodexProvider,
    readiness=PROBE,
    model_vendor="openai",
    pricing_vendor="openai",
    model_entries=MODEL_ENTRIES,
    pricing_entries=PRICING_ENTRIES,
    rates=RATES,
    transcript_format=TRANSCRIPTS,
    limits=CodexLimitsReader(),
    stream_factory=CodexAppServerTransport,
    testing=ProviderTestSupport(
        scripted_lines=scripted_lines,
        resume_failure_lines=resume_failure_lines,
        failure_lines=failure_lines,
        installed_mcp_servers=installed_mcp_servers,
        probe_fixture=probe_fixture,
    ),
)
