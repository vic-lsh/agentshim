"""Static, versioned per-model token prices, and the cost of a :class:`TokenUsage`.

The table is data: one :class:`ModelPricing` per ``(vendor, model)``, each
with the official page it was read from and the date it was checked, and a
table-level ``version`` and ``last_updated``. Prices are USD per million
tokens at the vendor's standard (non-batch, global, short-context) tier.

A lookup of a model the table does not know returns ``None``: an unknown
price is never read as zero. Callers pin or extend assumptions by passing
their own :class:`PricingTable` (see :meth:`PricingTable.with_entries` and
:meth:`PricingTable.from_dict`).

Updating the table is described in ``docs/pricing.md``.
"""

from __future__ import annotations

import datetime as dt
import math
import re
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, cast

from agentshim.core.usage import TokenUsage, TokenWeights

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

#: Vendor each agentshim provider bills through, when it bills per token.
#: Copilot bills premium requests and Gemini is not priced here, so both are absent.
PROVIDER_VENDORS: Mapping[str, str] = {
    "codex": "openai",
    "openai": "openai",
    "claude": "anthropic",
    "anthropic": "anthropic",
}

_TOKENS_PER_UNIT = 1_000_000
_DATE_SUFFIX = re.compile(r"-\d{8}$")
_BRACKET_SUFFIX = re.compile(r"\[[^\]]*\]$")

OPENAI_PRICING_URL = "https://developers.openai.com/api/docs/pricing"
ANTHROPIC_PRICING_URL = "https://platform.claude.com/docs/en/about-claude/pricing"


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


def vendor_and_model(provider: str, model: str) -> tuple[str, str] | None:
    """Resolve an agentshim provider and model name to a table key.

    Accepts a provider name (``codex``, ``claude``) or a vendor name, and a
    model with a ``vendor/`` prefix (opencode's ``provider/model``). Drops a
    trailing date snapshot (``claude-haiku-4-5-20251001``) and a bracketed
    variant (``claude-opus-5-5[1m]``). Returns ``None`` when the provider is
    not billed per token by a known vendor.
    """
    name = model.strip().lower()
    vendor = PROVIDER_VENDORS.get(provider.strip().lower())
    if "/" in name:
        prefix, _, rest = name.partition("/")
        vendor = PROVIDER_VENDORS.get(prefix, vendor)
        name = rest
    if vendor is None or not name:
        return None
    name = _BRACKET_SUFFIX.sub("", name)
    name = _DATE_SUFFIX.sub("", name)
    return (vendor, name)


def price_for(
    provider: str, model: str | None, table: PricingTable | None = None
) -> ModelPricing | None:
    """The price of *model* run through *provider*, or ``None`` when it is not known.

    Never falls back to another model's price or to zero.
    """
    if not model:
        return None
    key = vendor_and_model(provider, model)
    if key is None:
        return None
    return (table or DEFAULT_PRICING).get(*key)


def default_pricing() -> PricingTable:
    """The table agentshim ships (:data:`DEFAULT_PRICING`)."""
    return DEFAULT_PRICING


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


_CHECKED = dt.date(2026, 9, 27)
_OPENAI_NOTE = (
    "Short-context standard tier; the long-context rates on the same page are not modelled."
)
_OPENAI_NO_WRITE_NOTE = (
    "Short-context standard tier; the page lists no cache-write rate, so cache writes "
    "are priced at base input."
)


def _openai(
    model: str,
    input_usd: float,
    cache_read_usd: float,
    output_usd: float,
    cache_write_usd: float | None = None,
) -> ModelPricing:
    return ModelPricing(
        vendor="openai",
        model=model,
        input_usd=input_usd,
        cache_read_usd=cache_read_usd,
        cache_write_usd=input_usd if cache_write_usd is None else cache_write_usd,
        output_usd=output_usd,
        source=OPENAI_PRICING_URL,
        checked=_CHECKED,
        note=_OPENAI_NOTE if cache_write_usd is not None else _OPENAI_NO_WRITE_NOTE,
    )


def _anthropic(  # noqa: PLR0913 - one positional per price column
    model: str,
    input_usd: float,
    cache_write_usd: float,
    cache_write_1h_usd: float,
    cache_read_usd: float,
    output_usd: float,
) -> ModelPricing:
    return ModelPricing(
        vendor="anthropic",
        model=model,
        input_usd=input_usd,
        cache_read_usd=cache_read_usd,
        cache_write_usd=cache_write_usd,
        cache_write_1h_usd=cache_write_1h_usd,
        output_usd=output_usd,
        source=ANTHROPIC_PRICING_URL,
        checked=_CHECKED,
        note="Claude API first-party, global inference; thinking is billed as output.",
    )


#: The table agentshim ships. Update it as described in ``docs/pricing.md``.
DEFAULT_PRICING = PricingTable(
    version="2026-09-27",
    last_updated=_CHECKED,
    entries=(
        # OpenAI (Codex). Reasoning tokens are part of output and billed as output.
        _openai("gpt-6-astra", 10.00, 1.00, 50.00, cache_write_usd=12.50),
        _openai("gpt-6-sol", 2.00, 0.20, 10.00, cache_write_usd=2.50),
        _openai("gpt-6-luna", 0.10, 0.01, 0.50, cache_write_usd=0.125),
        _openai("gpt-5.6-sol", 4.00, 0.40, 20.00, cache_write_usd=5.00),
        _openai("gpt-5.6-terra", 2.00, 0.20, 12.00, cache_write_usd=2.50),
        _openai("gpt-5.6-luna", 0.20, 0.02, 1.20, cache_write_usd=0.25),
        _openai("gpt-5.5", 5.00, 0.50, 30.00),
        _openai("gpt-5.4", 2.50, 0.25, 15.00),
        _openai("gpt-5.4-mini", 0.75, 0.075, 4.50),
        _openai("gpt-5.4-nano", 0.20, 0.02, 1.25),
        _openai("gpt-5.3-codex", 1.75, 0.175, 14.00),
        _openai("gpt-5.2", 1.75, 0.175, 14.00),
        _openai("gpt-5.1", 1.25, 0.125, 10.00),
        _openai("gpt-5", 1.25, 0.125, 10.00),
        _openai("gpt-5-mini", 0.25, 0.025, 2.00),
        _openai("gpt-5-nano", 0.05, 0.005, 0.40),
        # Anthropic (Claude): input, 5m write, 1h write, cache read, output.
        _anthropic("claude-fable-5-1", 10.00, 12.50, 20.00, 0.25, 50.00),
        _anthropic("claude-fable-5", 10.00, 12.50, 20.00, 1.00, 50.00),
        _anthropic("claude-opus-5-5", 4.00, 5.00, 8.00, 0.20, 20.00),
        _anthropic("claude-opus-5", 5.00, 6.25, 10.00, 0.50, 25.00),
        _anthropic("claude-opus-4-8", 5.00, 6.25, 10.00, 0.50, 25.00),
        _anthropic("claude-sonnet-5", 2.00, 2.50, 4.00, 0.20, 10.00),
        _anthropic("claude-sonnet-4-6", 3.00, 3.75, 6.00, 0.30, 15.00),
        _anthropic("claude-haiku-4-5", 1.00, 1.25, 2.00, 0.10, 5.00),
    ),
)
