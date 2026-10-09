"""Durable record of which provider conversation a session was in.

A checkpoint is what a restarted process needs to continue a conversation: the
provider's id for it and the usage baseline of its last report. Where it is
stored is the caller's business, so the store is a protocol.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from .usage import ProviderUsage


@dataclass(frozen=True)
class Checkpoint:
    """A provider conversation a later process can resume.

    ``usage`` is the conversation's most recent report, kept so a provider
    whose usage is cumulative can still tell the next turn's increment.
    """

    conversation_id: str
    usage: ProviderUsage | None = None


class CheckpointStore(Protocol):
    """Where a session keeps its checkpoint between processes."""

    def load(self, key: str) -> Checkpoint | None:
        """Return the checkpoint saved under *key*, or ``None``."""
        ...

    def save(self, key: str, checkpoint: Checkpoint) -> None:
        """Replace whatever is saved under *key*."""
        ...

    def clear(self, key: str) -> None:
        """Forget *key*; clearing a key that holds nothing is not an error."""
        ...


class InMemoryCheckpointStore:
    """A ``CheckpointStore`` that lives and dies with the process."""

    def __init__(self) -> None:
        """Start empty."""
        self._lock = threading.Lock()
        self._items: dict[str, Checkpoint] = {}

    def load(self, key: str) -> Checkpoint | None:
        """Return the checkpoint saved under *key*, or ``None``."""
        with self._lock:
            return self._items.get(key)

    def save(self, key: str, checkpoint: Checkpoint) -> None:
        """Replace whatever is saved under *key*."""
        with self._lock:
            self._items[key] = checkpoint

    def clear(self, key: str) -> None:
        """Forget *key*."""
        with self._lock:
            self._items.pop(key, None)
