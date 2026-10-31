"""Doubles for the rate refresh: a scripted source and an in-memory store."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agentshim.rates import RateFetchError, RateSnapshot

if TYPE_CHECKING:
    from agentshim.rates import RateEndpoint


class FakeRateSource:
    """A ``RateSource`` that answers from a script instead of the network.

    Each call to :meth:`fetch` consumes the next outcome: bytes are returned,
    a :class:`RateFetchError` is raised. When the script runs out every call
    fails, as a dead network would.
    """

    def __init__(self, *outcomes: bytes | RateFetchError) -> None:
        """Script the outcomes, first call first."""
        self._outcomes = list(outcomes)
        #: The endpoint of every call, in order.
        self.requests: list[RateEndpoint] = []

    def fetch(self, endpoint: RateEndpoint) -> bytes:
        """Return or raise the next scripted outcome."""
        self.requests.append(endpoint)
        if not self._outcomes:
            msg = f"no outcome scripted for {endpoint.url}"
            raise RateFetchError(msg)
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, RateFetchError):
            raise outcome
        return outcome


class FakeRateStore:
    """A ``RateStore`` that keeps its snapshot in memory."""

    def __init__(self, snapshot: RateSnapshot | None = None) -> None:
        """Start holding *snapshot*, or nothing."""
        self.snapshot = snapshot
        #: Every snapshot written, in order.
        self.writes: list[RateSnapshot] = []

    def read(self) -> RateSnapshot | None:
        """The snapshot last written, or the one it started with."""
        return self.snapshot

    def write(self, snapshot: RateSnapshot) -> None:
        """Replace the snapshot, remembering the write."""
        self.snapshot = snapshot
        self.writes.append(snapshot)
