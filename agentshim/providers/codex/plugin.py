"""Codex's complete catalog declaration."""

from __future__ import annotations

from agentshim.core.plugin import ProviderPlugin, ProviderTestSupport

from .app_server import CodexAppServerTransport
from .model_data import MODEL_ENTRIES, PRICING_ENTRIES
from .probe import PROBE
from .provider import PROFILE, CodexProvider
from .scripted import failure_lines, resume_failure_lines, scripted_lines
from .testing import installed_mcp_servers, probe_fixture

PLUGIN = ProviderPlugin(
    name="codex",
    profile=PROFILE,
    provider_factory=CodexProvider,
    probe=PROBE,
    model_vendor="openai",
    pricing_vendor="openai",
    model_entries=MODEL_ENTRIES,
    pricing_entries=PRICING_ENTRIES,
    stream_factory=CodexAppServerTransport,
    testing=ProviderTestSupport(
        scripted_lines=scripted_lines,
        resume_failure_lines=resume_failure_lines,
        failure_lines=failure_lines,
        installed_mcp_servers=installed_mcp_servers,
        probe_fixture=probe_fixture,
    ),
)
