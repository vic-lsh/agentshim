"""Claude Code's ``rate_limit_event`` frame, read as provider-neutral events.

The frame is ``{"type":"rate_limit_event","rate_limit_info":{...}}`` and the
info carries a ``status`` (``allowed``, ``allowed_warning`` or ``rejected``),
the ``rateLimitType`` the status is about, and ``unifiedWindows``: one entry
per window with a ``utilization`` fraction and a ``resetsAt`` in epoch seconds.
Every field is optional here, because a window the frame does not describe is
unknown rather than empty.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from agentshim.core.events import RateLimitStatus

if TYPE_CHECKING:
    from collections.abc import Mapping

_ALLOWED = frozenset({"allowed", "allowed_warning"})


@dataclass(frozen=True)
class RateLimitFrame:
    """``{"type":"rate_limit_event",...}``: the windows the CLI just reported."""

    statuses: tuple[RateLimitStatus, ...]


def parse_rate_limit(data: Mapping[str, Any]) -> RateLimitFrame | None:
    """Read a ``rate_limit_event`` object, or ``None`` when it has no usable info."""
    info = _mapping(data.get("rate_limit_info"))
    if info is None:
        return None
    status = info.get("status")
    named = info.get("rateLimitType")
    named = named if isinstance(named, str) and named else None
    windows = _mapping(info.get("unifiedWindows")) or {}
    statuses = [
        RateLimitStatus(
            window=name,
            used_fraction=_number(_window_field(window, "utilization")),
            resets_at=_number(_window_field(window, "resetsAt")),
            exhausted=_exhausted(status) if name == named else None,
            raw=info,
        )
        for name, window in windows.items()
    ]
    if named is not None and named not in windows:
        statuses.append(
            RateLimitStatus(
                window=named,
                used_fraction=None,
                resets_at=_number(info.get("resetsAt")),
                exhausted=_exhausted(status),
                raw=info,
            )
        )
    return RateLimitFrame(tuple(statuses)) if statuses else None


def _exhausted(status: object) -> bool | None:
    if status == "rejected":
        return True
    return False if status in _ALLOWED else None


def _window_field(window: object, key: str) -> object:
    mapping = _mapping(window)
    return None if mapping is None else mapping.get(key)


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _mapping(value: object) -> Mapping[str, Any] | None:
    return cast("Mapping[str, Any]", value) if isinstance(value, dict) else None
