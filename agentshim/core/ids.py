"""Identifier allocation as an injected dependency."""

from __future__ import annotations

import uuid
from typing import Protocol


class IdAllocator(Protocol):
    """Hands out identifiers that are unique within one run."""

    def new_id(self, prefix: str) -> str:
        """Return a fresh identifier that starts with *prefix*."""
        ...


class RandomIds:
    """``<prefix>-<uuid4 hex>``: unique across processes and runs."""

    def new_id(self, prefix: str) -> str:
        """Return ``prefix`` joined to a random UUID."""
        return f"{prefix}-{uuid.uuid4().hex}"
