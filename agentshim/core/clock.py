"""Time as an injected dependency, so waits can be cancelled and simulated."""

from __future__ import annotations

import threading
import time
from typing import Protocol


class StopSignal:
    """A cancellation flag that a wait can observe.

    A thin wrapper over ``threading.Event`` so a ``Clock`` implementation, real
    or fake, has one small type to honor.
    """

    def __init__(self) -> None:
        """Start unset."""
        self._event = threading.Event()

    def set(self) -> None:
        """Mark the signal; every current and later wait returns early."""
        self._event.set()

    def is_set(self) -> bool:
        """Whether ``set`` has been called."""
        return self._event.is_set()

    def wait(self, timeout: float) -> bool:
        """Block up to *timeout* seconds; return whether the signal is set."""
        return self._event.wait(timeout)


class Clock(Protocol):
    """Monotonic time and cancellable waiting."""

    def monotonic(self) -> float:
        """Seconds on a clock that never goes backwards."""
        ...

    def wait(self, seconds: float, stop: StopSignal | None = None) -> bool:
        """Wait *seconds*, or until *stop* is set; return ``True`` if stopped early."""
        ...


class SystemClock:
    """The real clock."""

    def monotonic(self) -> float:
        """Return ``time.monotonic()``."""
        return time.monotonic()

    def wait(self, seconds: float, stop: StopSignal | None = None) -> bool:
        """Sleep up to *seconds*; wake early and return ``True`` when *stop* is set."""
        seconds = max(seconds, 0.0)
        if stop is None:
            if seconds:
                time.sleep(seconds)
            return False
        return stop.wait(seconds)
