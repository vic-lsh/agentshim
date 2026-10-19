"""Provider-independent model facts and tables."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Iterable


@dataclass(frozen=True)
class ModelSpec:
    """One model's context window, with where and when it was read.

    Attributes:
        vendor: Who serves the model (``"openai"``, ``"anthropic"``).
        model: The model id as the vendor names it.
        context_window: The model's context window in tokens (input and
            output together), at the vendor's default for that id.
        source: URL of the official page the number was read from.
        checked: Date the number was last checked against ``source``.
        display_name: A readable name for the model (``"Opus 5.5"``), or
            ``None`` for a caller-built entry that has none.
        family: The model line the model belongs to (``"Claude"``, ``"GPT"``,
            ``"Gemini"``); set together with ``display_name``.
        tier: The model's rank within its family, ``1`` the smallest and
            cheapest; versions of one tier share a rank. ``None`` for a model
            that sits on no tier ladder.
        tier_name: The tier's readable name (``"Opus"``); set together with
            ``tier``.
        generation_id: A short stable token naming the model's generation
            within its vendor (``"5.5"``); every model of one generation
            shares it, and no two generations of a vendor do.
        generation: The generation as people call it (``"Claude 5.5"``,
            ``"GPT-6"``); set together with ``generation_id``.
        generation_rank: The generation's rank among its vendor's, ``1`` the
            newest; set together with ``generation_id``. ``ModelTable``
            checks that one vendor's generations have distinct ranks.
    """

    vendor: str
    model: str
    context_window: int
    source: str
    checked: dt.date
    display_name: str | None = None
    family: str | None = None
    tier: int | None = None
    tier_name: str | None = None
    generation_id: str | None = None
    generation: str | None = None
    generation_rank: int | None = None

    def __post_init__(self) -> None:
        """Reject a missing name, source or date, and a window that is not a positive int."""
        for name in ("vendor", "model"):
            value: object = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                msg = f"{name} must be a non-empty string"
                raise ValueError(msg)
        source = cast("object", self.source)
        if not isinstance(source, str) or not source.startswith("https://"):
            msg = f"{self.vendor}/{self.model}: source must be an https:// URL"
            raise ValueError(msg)
        checked = cast("object", self.checked)
        if not isinstance(checked, dt.date) or isinstance(checked, dt.datetime):
            msg = f"{self.vendor}/{self.model}: checked must be a datetime.date"
            raise TypeError(msg)
        window = cast("object", self.context_window)
        if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
            msg = f"{self.vendor}/{self.model}: context_window must be a positive int"
            raise ValueError(msg)
        self._check_presentation()
        self._check_generation()

    def _check_generation(self) -> None:
        """Reject a half-set generation triple, a blank name, and a rank that is not a positive int."""
        parts = (self.generation_id, self.generation, self.generation_rank)
        if any(part is None for part in parts) and any(part is not None for part in parts):
            msg = (
                f"{self.vendor}/{self.model}: "
                "generation_id, generation and generation_rank are set together"
            )
            raise ValueError(msg)
        for name in ("generation_id", "generation"):
            value: object = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                msg = f"{self.vendor}/{self.model}: {name} must be a non-empty string"
                raise ValueError(msg)
        rank = cast("object", self.generation_rank)
        if rank is not None and (isinstance(rank, bool) or not isinstance(rank, int) or rank < 1):
            msg = f"{self.vendor}/{self.model}: generation_rank must be a positive int"
            raise ValueError(msg)

    def _check_presentation(self) -> None:
        """Reject a half-set name pair or tier pair, and a tier that is not a positive int."""
        for first, second in (("display_name", "family"), ("tier", "tier_name")):
            if (getattr(self, first) is None) != (getattr(self, second) is None):
                msg = f"{self.vendor}/{self.model}: {first} and {second} are set together"
                raise ValueError(msg)
        for name in ("display_name", "family", "tier_name"):
            value: object = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                msg = f"{self.vendor}/{self.model}: {name} must be a non-empty string"
                raise ValueError(msg)
        tier = cast("object", self.tier)
        if tier is not None and (isinstance(tier, bool) or not isinstance(tier, int) or tier < 1):
            msg = f"{self.vendor}/{self.model}: tier must be a positive int"
            raise ValueError(msg)
        if self.tier is not None and self.family is None:
            msg = f"{self.vendor}/{self.model}: tier needs a family"
            raise ValueError(msg)

    @property
    def key(self) -> tuple[str, str]:
        """The ``(vendor, model)`` this entry describes."""
        return (self.vendor, self.model)


@dataclass(frozen=True)
class ModelTable:
    """A versioned set of :class:`ModelSpec` entries.

    Attributes:
        version: Names this set of facts; change it whenever a fact changes.
        last_updated: Date the table as a whole was last updated. No entry
            may have been checked later.
        entries: One entry per ``(vendor, model)``.
    """

    version: str
    last_updated: dt.date
    entries: tuple[ModelSpec, ...] = field(default=())

    def __post_init__(self) -> None:
        """Reject a missing version, duplicate keys, and entries checked after ``last_updated``."""
        version = cast("object", self.version)
        if not isinstance(version, str) or not version.strip():
            msg = "version must be a non-empty string"
            raise ValueError(msg)
        object.__setattr__(self, "entries", tuple(self.entries))
        seen: set[tuple[str, str]] = set()
        for entry in self.entries:
            if not isinstance(cast("object", entry), ModelSpec):
                msg = f"entries must be ModelSpec, got {entry!r}"
                raise TypeError(msg)
            if entry.key in seen:
                msg = f"duplicate model entry for {entry.vendor}/{entry.model}"
                raise ValueError(msg)
            seen.add(entry.key)
            if entry.checked > self.last_updated:
                msg = (
                    f"{entry.vendor}/{entry.model} was checked {entry.checked}, "
                    f"after the table's last_updated {self.last_updated}"
                )
                raise ValueError(msg)
        self._check_generations()

    def _check_generations(self) -> None:
        """Reject a vendor whose generations disagree on a name or rank, or share a rank."""
        by_id: dict[tuple[str, str], tuple[str | None, int | None]] = {}
        by_rank: dict[tuple[str, int], str] = {}
        for entry in self.entries:
            if entry.generation_id is None or entry.generation_rank is None:
                continue
            known = by_id.setdefault(
                (entry.vendor, entry.generation_id), (entry.generation, entry.generation_rank)
            )
            if known != (entry.generation, entry.generation_rank):
                msg = (
                    f"{entry.vendor}/{entry.model}: generation {entry.generation_id!r} "
                    "has another name or rank on another model"
                )
                raise ValueError(msg)
            holder = by_rank.setdefault((entry.vendor, entry.generation_rank), entry.generation_id)
            if holder != entry.generation_id:
                msg = (
                    f"{entry.vendor}: generations {holder!r} and {entry.generation_id!r} "
                    f"share rank {entry.generation_rank}"
                )
                raise ValueError(msg)

    def get(self, vendor: str, model: str) -> ModelSpec | None:
        """The entry for exactly ``(vendor, model)``, or ``None``."""
        for entry in self.entries:
            if entry.key == (vendor, model):
                return entry
        return None

    def with_entries(
        self,
        entries: Iterable[ModelSpec],
        *,
        version: str,
        last_updated: dt.date | None = None,
    ) -> ModelTable:
        """A new table with *entries* added, replacing any with the same key."""
        added = tuple(entries)
        by_key = {entry.key: entry for entry in self.entries}
        by_key.update({entry.key: entry for entry in added})
        latest = max([self.last_updated, *(entry.checked for entry in added)])
        return replace(
            self,
            version=version,
            last_updated=last_updated or latest,
            entries=tuple(by_key.values()),
        )
