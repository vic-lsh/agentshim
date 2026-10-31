"""The speed Codex bills a request at."""

from __future__ import annotations

from agentshim.core.usage import UsageSpeed


def speed_of_tier(service_tier: str | None) -> UsageSpeed:
    """The speed Codex's ``service_tier`` bills at: ``priority`` is fast; unset, default and flex are standard."""
    if service_tier in ("priority", "fast"):
        return UsageSpeed.FAST
    if service_tier == "ultrafast":
        return UsageSpeed.ULTRAFAST
    return UsageSpeed.STANDARD
