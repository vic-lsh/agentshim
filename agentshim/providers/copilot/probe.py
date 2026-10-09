"""Copilot readiness: a version only; authentication is not checkable."""

from __future__ import annotations

from agentshim.core.status import ProbeSpec

PROBE = ProbeSpec(
    auth=None,
    auth_note="Copilot CLI has no authentication status command",
)
