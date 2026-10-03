"""What a turn did with skills, folded from its events.

``SkillTracker`` is an ordinary event handler, so the summary on
``TurnResult.skills`` and any caller that watches the stream itself read the
same events: there is one source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .events import SkillInvoked, SkillsDiscovered
from .profile import SkillSignal

if TYPE_CHECKING:
    from .events import AgentEvent
    from .profile import ProviderProfile


@dataclass(frozen=True)
class SkillSummary:
    """Skills a turn (or several) offered and loaded.

    ``None`` means unknown: the provider's stream carries no such signal, or
    did not carry it this time. It never means zero. An empty tuple means the
    signal was available and nothing matched.
    """

    discovered: tuple[str, ...] | None = None
    invocations: tuple[SkillInvoked, ...] | None = None

    @property
    def invoked(self) -> tuple[str, ...] | None:
        """Distinct invoked skill names, in order of first use."""
        if self.invocations is None:
            return None
        return tuple(dict.fromkeys(event.name for event in self.invocations))

    @property
    def invocation_count(self) -> int | None:
        """How many loads happened, or ``None`` when unknown."""
        if self.invocations is None:
            return None
        return len(self.invocations)


class SkillTracker:
    """Event handler that folds skill events into a ``SkillSummary``.

    Feed it one turn or many; the summary covers everything it has seen.
    """

    def __init__(self, profile: ProviderProfile) -> None:
        """Track skills for a provider with *profile*'s declared signals."""
        self._invocation_known = profile.skill_invocation is not SkillSignal.NONE
        self._discovered: list[str] | None = None
        self._invocations: list[SkillInvoked] = []

    def on_event(self, event: AgentEvent) -> None:
        """Record a skill event; ignore every other event."""
        if isinstance(event, SkillsDiscovered):
            known = self._discovered if self._discovered is not None else []
            known.extend(name for name in event.names if name not in known)
            self._discovered = known
        elif isinstance(event, SkillInvoked):
            self._invocations.append(event)

    def summary(self) -> SkillSummary:
        """Return what has been seen so far."""
        discovered = tuple(self._discovered) if self._discovered is not None else None
        invocations: tuple[SkillInvoked, ...] | None = tuple(self._invocations)
        if not self._invocation_known and not self._invocations:
            invocations = None
        return SkillSummary(discovered=discovered, invocations=invocations)
