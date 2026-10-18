"""Provider-independent pricing facts, tables, and cost calculation."""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, cast

from .usage import TokenUsage, TokenWeights

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

_TOKENS_PER_UNIT = 1_000_000


@dataclass(frozen=True)
class ModelPricing:
    """One model's USD price per million tokens, with where and when it was read.

    Attributes:
        vendor: Who bills for the model (``"openai"``, ``"anthropic"``).
        model: The model id as the vendor names it.
        input_usd: Base (uncached) input.
        cache_read_usd: Input read from the prompt cache.
        cache_write_usd: Input written to the prompt cache (Anthropic's
            five-minute TTL). A vendor with no cache-write surcharge lists its
            base input rate here.
        output_usd: Output, reasoning included.
        source: URL of the official pricing page the numbers were read from.
        checked: Date the numbers were last checked against ``source``.
        cache_write_1h_usd: One-hour-TTL cache writes; ``None`` when the
            vendor has no such tier (``cache_write_usd`` applies).
        reasoning_usd: Reasoning output, when billed apart from output;
            ``None`` means reasoning is billed as output.
        estimated: Whether any number is an estimate rather than a published
            price; ``note`` must then state the basis.
        note: Caveats, such as a long-context surcharge the table leaves out.
    """

    vendor: str
    model: str
    input_usd: float
    cache_read_usd: float
    cache_write_usd: float
    output_usd: float
    source: str
    checked: dt.date
    cache_write_1h_usd: float | None = None
    reasoning_usd: float | None = None
    estimated: bool = False
    note: str = ""

    def __post_init__(self) -> None:
        """Reject a missing source or date, and any price that is not a finite non-negative number."""
        for name in ("vendor", "model"):
            value: object = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                msg = f"{name} must be a non-empty string"
                raise ValueError(msg)
        source = cast("object", self.source)
        if not isinstance(source, str) or not source.startswith("https://"):
            msg = f"{self.vendor}/{self.model}: source must be an https:// URL"
            raise ValueError(msg)
        if not _is_date(self.checked):
            msg = f"{self.vendor}/{self.model}: checked must be a datetime.date"
            raise TypeError(msg)
        for name in ("input_usd", "cache_read_usd", "cache_write_usd", "output_usd"):
            _check_price(self, name, getattr(self, name))
        for name in ("cache_write_1h_usd", "reasoning_usd"):
            optional: object = getattr(self, name)
            if optional is not None:
                _check_price(self, name, optional)
        if self.input_usd <= 0:
            msg = f"{self.vendor}/{self.model}: input_usd must be positive"
            raise ValueError(msg)
        if self.estimated and not self.note.strip():
            msg = f"{self.vendor}/{self.model}: an estimated entry must state its basis in note"
            raise ValueError(msg)

    @property
    def key(self) -> tuple[str, str]:
        """The ``(vendor, model)`` this entry prices."""
        return (self.vendor, self.model)

    def usd_weights(self) -> TokenWeights:
        """Weights in USD per million tokens, for :meth:`TokenUsage.weighted_total`."""
        return TokenWeights(
            uncached_input=self.input_usd,
            cache_read_input=self.cache_read_usd,
            cache_write_input=self.cache_write_usd,
            cache_write_1h_input=self.cache_write_1h_usd,
            output=self.output_usd,
            reasoning_output=self.reasoning_usd,
        )

    def relative_weights(self) -> TokenWeights:
        """Weights as multiples of the base input rate (uncached input is 1.0)."""
        base = self.input_usd

        def rel(value: float | None) -> float | None:
            return None if value is None else value / base

        return TokenWeights(
            uncached_input=1.0,
            cache_read_input=self.cache_read_usd / base,
            cache_write_input=self.cache_write_usd / base,
            cache_write_1h_input=rel(self.cache_write_1h_usd),
            output=self.output_usd / base,
            reasoning_output=rel(self.reasoning_usd),
        )

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serializable mapping that :meth:`from_dict` reads back."""
        return {
            "vendor": self.vendor,
            "model": self.model,
            "input_usd": self.input_usd,
            "cache_read_usd": self.cache_read_usd,
            "cache_write_usd": self.cache_write_usd,
            "cache_write_1h_usd": self.cache_write_1h_usd,
            "output_usd": self.output_usd,
            "reasoning_usd": self.reasoning_usd,
            "source": self.source,
            "checked": self.checked.isoformat(),
            "estimated": self.estimated,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ModelPricing:
        """Build an entry from a mapping; ``checked`` is an ISO date string or a date.

        Raises:
            ValueError, TypeError: A required field is missing or invalid.
        """
        missing = [
            name
            for name in (
                "vendor",
                "model",
                "input_usd",
                "cache_read_usd",
                "cache_write_usd",
                "output_usd",
                "source",
                "checked",
            )
            if data.get(name) is None
        ]
        if missing:
            msg = f"pricing entry is missing {', '.join(missing)}"
            raise ValueError(msg)
        return cls(
            vendor=str(data["vendor"]),
            model=str(data["model"]),
            input_usd=float(data["input_usd"]),
            cache_read_usd=float(data["cache_read_usd"]),
            cache_write_usd=float(data["cache_write_usd"]),
            output_usd=float(data["output_usd"]),
            source=str(data["source"]),
            checked=_date(data["checked"], "checked"),
            cache_write_1h_usd=_optional_float(data.get("cache_write_1h_usd")),
            reasoning_usd=_optional_float(data.get("reasoning_usd")),
            estimated=bool(data.get("estimated", False)),
            note=str(data.get("note") or ""),
        )


@dataclass(frozen=True)
class PricingTable:
    """A versioned set of :class:`ModelPricing` entries.

    Attributes:
        version: Names this set of prices; change it whenever a price changes.
        last_updated: Date the table as a whole was last updated. No entry
            may have been checked later.
        entries: One entry per ``(vendor, model)``.
    """

    version: str
    last_updated: dt.date
    entries: tuple[ModelPricing, ...] = field(default=())

    def __post_init__(self) -> None:
        """Reject a missing version or date, duplicate keys, and entries checked after ``last_updated``."""
        version = cast("object", self.version)
        if not isinstance(version, str) or not version.strip():
            msg = "version must be a non-empty string"
            raise ValueError(msg)
        if not _is_date(self.last_updated):
            msg = "last_updated must be a datetime.date"
            raise TypeError(msg)
        object.__setattr__(self, "entries", tuple(self.entries))
        seen: set[tuple[str, str]] = set()
        for entry in self.entries:
            if not isinstance(cast("object", entry), ModelPricing):
                msg = f"entries must be ModelPricing, got {entry!r}"
                raise TypeError(msg)
            if entry.key in seen:
                msg = f"duplicate pricing entry for {entry.vendor}/{entry.model}"
                raise ValueError(msg)
            seen.add(entry.key)
            if entry.checked > self.last_updated:
                msg = (
                    f"{entry.vendor}/{entry.model} was checked {entry.checked}, "
                    f"after the table's last_updated {self.last_updated}"
                )
                raise ValueError(msg)

    def get(self, vendor: str, model: str) -> ModelPricing | None:
        """The entry for exactly ``(vendor, model)``, or ``None``."""
        for entry in self.entries:
            if entry.key == (vendor, model):
                return entry
        return None

    def with_entries(
        self,
        entries: Iterable[ModelPricing],
        *,
        version: str,
        last_updated: dt.date | None = None,
    ) -> PricingTable:
        """A new table with *entries* added, replacing any with the same key.

        A changed table is a different set of prices, so it needs its own
        ``version``. ``last_updated`` defaults to the latest of this table's
        date and the new entries' ``checked`` dates.
        """
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

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serializable mapping that :meth:`from_dict` reads back."""
        return {
            "version": self.version,
            "last_updated": self.last_updated.isoformat(),
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PricingTable:
        """Build a table from a mapping such as a parsed JSON or TOML file.

        Raises:
            ValueError, TypeError: ``version``, ``last_updated`` or an entry
                field is missing or invalid.
        """
        if data.get("version") is None or data.get("last_updated") is None:
            msg = "pricing table needs version and last_updated"
            raise ValueError(msg)
        raw_entries: object = data.get("entries") or []
        if not isinstance(raw_entries, list):
            msg = "pricing table entries must be a list"
            raise TypeError(msg)
        rows = cast("list[Mapping[str, Any]]", raw_entries)
        return cls(
            version=str(data["version"]),
            last_updated=_date(data["last_updated"], "last_updated"),
            entries=tuple(ModelPricing.from_dict(row) for row in rows),
        )


def cost_usd(usage: TokenUsage, pricing: ModelPricing) -> float:
    """USD cost of *usage* at *pricing*.

    Every token class is priced once: uncached input, cache reads, cache
    writes by TTL, and output (reasoning at ``reasoning_usd`` when set).
    """
    return usage.weighted_total(pricing.usd_weights()) / _TOKENS_PER_UNIT


def _check_price(entry: ModelPricing, name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        msg = f"{entry.vendor}/{entry.model}: {name} must be a number, got {value!r}"
        raise TypeError(msg)
    if not math.isfinite(value) or value < 0:
        msg = f"{entry.vendor}/{entry.model}: {name} must be finite and non-negative"
        raise ValueError(msg)


def _is_date(value: object) -> bool:
    return isinstance(value, dt.date) and not isinstance(value, dt.datetime)


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)  # pyright: ignore[reportArgumentType]


def _date(value: object, name: str) -> dt.date:
    if isinstance(value, dt.datetime):
        msg = f"{name} must be a date, not a datetime"
        raise TypeError(msg)
    if isinstance(value, dt.date):
        return value
    if isinstance(value, str):
        return dt.date.fromisoformat(value)
    msg = f"{name} must be an ISO date string or a date, got {value!r}"
    raise TypeError(msg)
