"""Deterministic time and identifiers."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentshim.core.clock import StopSignal


class FakeClock:
    """A ``Clock`` whose waits take no real time.

    ``monotonic`` starts at 0. ``wait`` advances it by the requested seconds
    at once, unless *stop* is already set, and records the request in
    ``waits``.
    """

    def __init__(self) -> None:
        """Start at time zero with no waits recorded."""
        self._now = 0.0
        self.waits: list[float] = []

    def monotonic(self) -> float:
        """Return the simulated time."""
        return self._now

    def advance(self, seconds: float) -> None:
        """Move simulated time forward without recording a wait."""
        self._now += max(seconds, 0.0)

    def wait(self, seconds: float, stop: StopSignal | None = None) -> bool:
        """Advance by *seconds* and return ``False``; return ``True`` at once if stopped."""
        self.waits.append(seconds)
        if stop is not None and stop.is_set():
            return True
        self.advance(seconds)
        return False


class SequentialIds:
    """An ``IdAllocator`` that numbers ids per prefix: ``turn-1``, ``turn-2``, ..."""

    def __init__(self) -> None:
        """Start every prefix at 1."""
        self._counts: dict[str, int] = {}

    def new_id(self, prefix: str) -> str:
        """Return ``<prefix>-<n>`` with *n* counting up from 1 for this prefix."""
        count = self._counts.get(prefix, 0) + 1
        self._counts[prefix] = count
        return f"{prefix}-{count}"
