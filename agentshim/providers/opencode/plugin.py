"""opencode's complete catalog declaration."""

from __future__ import annotations

from agentshim.core.plugin import ProviderPlugin, ProviderTestSupport

from .model_data import MODEL_ENTRIES
from .probe import PROBE
from .provider import PROFILE, OpencodeProvider
from .scripted import resume_failure_lines, scripted_lines
from .testing import installed_mcp_servers, probe_fixture

PLUGIN = ProviderPlugin(
    name="opencode",
    profile=PROFILE,
    provider_factory=OpencodeProvider,
    probe=PROBE,
    model_aliases=(("google-vertex", "google"),),
    model_entries=MODEL_ENTRIES,
    testing=ProviderTestSupport(
        scripted_lines=scripted_lines,
        resume_failure_lines=resume_failure_lines,
        installed_mcp_servers=installed_mcp_servers,
        probe_fixture=probe_fixture,
    ),
)
