"""Codex's ``account/rateLimits/updated`` notification as provider-neutral events."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agentshim.core.events import RateLimitStatus

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .protocol import RateLimitSnapshot, RateLimitWindow


def rate_limit_statuses(snapshot: RateLimitSnapshot) -> list[RateLimitStatus]:
    """One status per window the snapshot reports.

    ``rateLimitReachedType`` or ``spendControlReached`` marks the limit as
    reached, and a snapshot with neither says it is not. A snapshot that
    reports no window at all but does say a limit was reached still yields one
    window-less status, so the signal is not lost; one with neither yields
    nothing.
    """
    raw = snapshot.to_wire()
    reached = snapshot.rate_limit_reached_type is not None or bool(snapshot.spend_control_reached)
    limit = snapshot.limit_id or snapshot.limit_name
    windows = (("primary", snapshot.primary), ("secondary", snapshot.secondary))
    statuses = [
        _status(name, window, limit=limit, reached=reached, raw=raw)
        for name, window in windows
        if window is not None
    ]
    if not statuses and reached:
        statuses.append(
            RateLimitStatus(None, None, None, limit=limit, exhausted=True, raw=raw),
        )
    return statuses


def _status(
    name: str,
    window: RateLimitWindow,
    *,
    limit: str | None,
    reached: bool,
    raw: Mapping[str, object],
) -> RateLimitStatus:
    return RateLimitStatus(
        window=name,
        used_fraction=window.used_percent / 100,
        resets_at=None if window.resets_at is None else float(window.resets_at),
        limit=limit,
        window_minutes=window.window_duration_mins,
        exhausted=reached,
        raw=raw,
    )
