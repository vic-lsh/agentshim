"""Flat per-model rates for estimating what tokens would cost at list prices.

This is the second, separate price model. :class:`~agentshim.core.pricing.PricingTable`
holds the pinned prices a turn's recorded cost is computed from, keyed by
``(vendor, model)``, and it refuses a zero input rate. A :class:`RateTable`
holds rates read from a document such as LiteLLM's price file, keyed by one
flat name, and it allows a free model. The two share no type: :func:`price_tokens`
takes :class:`Rates`, while ``cost_usd`` and ``price_for`` take
:class:`~agentshim.core.pricing.ModelPricing`, so a fetched rate cannot reach a
recorded cost without a visible conversion.
"""

from __future__ import annotations

import datetime as dt
import math
import re
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, cast

from .errors import AgentShimError

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .usage import TokenUsage

_TOKENS_PER_UNIT = 1_000_000
_DATE_SUFFIX = re.compile(r"-\d{8}$")
_BRACKET_SUFFIX = re.compile(r"\[[^\]]*\]$")


class RateDocumentError(AgentShimError):
    """A rate document is not JSON, or holds no model with usable rates."""


class UsageSpeed(str, Enum):
    """The service speed a request was billed at."""

    STANDARD = "standard"
    FAST = "fast"
    ULTRAFAST = "ultrafast"


@dataclass(frozen=True)
class Rates:
    """USD per million tokens for one model at one speed.

    A free model has zero rates, which is why this type allows them and
    :class:`~agentshim.core.pricing.ModelPricing` does not.

    Attributes:
        input_usd: Uncached input.
        output_usd: Output, reasoning included.
        cache_read_usd: Input read from the prompt cache.
        cache_write_usd: Input written to the prompt cache.
        cache_write_1h_usd: One-hour-TTL cache writes; ``None`` when the
            document lists no such rate (``cache_write_usd`` then applies).
    """

    input_usd: float
    output_usd: float
    cache_read_usd: float
    cache_write_usd: float
    cache_write_1h_usd: float | None = None

    def __post_init__(self) -> None:
        """Reject a rate that is not a finite, non-negative number."""
        for name in ("input_usd", "output_usd", "cache_read_usd", "cache_write_usd"):
            _check(name, getattr(self, name))
        if self.cache_write_1h_usd is not None:
            _check("cache_write_1h_usd", self.cache_write_1h_usd)

    def scaled(self, multiple: float) -> Rates:
        """These rates times *multiple*, such as a fast tier priced at twice standard."""
        write_1h = self.cache_write_1h_usd
        return Rates(
            input_usd=self.input_usd * multiple,
            output_usd=self.output_usd * multiple,
            cache_read_usd=self.cache_read_usd * multiple,
            cache_write_usd=self.cache_write_usd * multiple,
            cache_write_1h_usd=None if write_1h is None else write_1h * multiple,
        )


@dataclass(frozen=True)
class RateEntry:
    """One model's rates at each speed it has a price for.

    Attributes:
        name: The normalized flat name (see :func:`normalize_name`).
        standard: The standard tier.
        fast: The fast tier, when the document lists one.
        ultrafast: The ultrafast tier, when the document lists one.
    """

    name: str
    standard: Rates
    fast: Rates | None = None
    ultrafast: Rates | None = None

    def __post_init__(self) -> None:
        """Reject a name that is not already normalized."""
        if not self.name or self.name != normalize_name(self.name):
            msg = f"rate entry name must be non-empty, trimmed and lowercase, got {self.name!r}"
            raise ValueError(msg)

    def at(self, speed: UsageSpeed) -> Rates | None:
        """The rates at *speed*, or ``None`` when the document lists none for it.

        Never falls back to another tier: whether a request at an unlisted
        speed is billed as standard is the caller's decision.
        """
        if speed is UsageSpeed.STANDARD:
            return self.standard
        return self.fast if speed is UsageSpeed.FAST else self.ultrafast


@dataclass(frozen=True)
class RateTable:
    """Rates by flat model name, read on one day.

    Attributes:
        checked_on: The day the document these rates came from was read.
        entries: Each entry under its own normalized name. A bare alias of a
            vendor-prefixed name is an entry too, when its sources agree.
    """

    checked_on: dt.date
    entries: Mapping[str, RateEntry]

    def __post_init__(self) -> None:
        """Freeze the entries and check each is filed under its own name."""
        checked_on = cast("object", self.checked_on)
        if not isinstance(checked_on, dt.date) or isinstance(checked_on, dt.datetime):
            msg = "checked_on must be a datetime.date"
            raise TypeError(msg)
        for key, entry in self.entries.items():
            if key != entry.name:
                msg = f"rate entry {entry.name!r} is filed under {key!r}"
                raise ValueError(msg)
        object.__setattr__(self, "entries", MappingProxyType(dict(self.entries)))

    def __len__(self) -> int:
        """How many names this table knows."""
        return len(self.entries)

    def lookup(self, name: str) -> RateEntry | None:
        """The entry for *name*, ignoring case and surrounding space, or ``None``."""
        return self.entries.get(normalize_name(name))


@dataclass(frozen=True)
class CostBreakdown:
    """What a set of tokens costs, split by the class of token.

    Attributes:
        input_usd: Uncached input.
        cache_read_usd: Cache reads.
        cache_write_usd: Cache writes of either TTL.
        output_usd: Output, reasoning included.
    """

    input_usd: float
    cache_read_usd: float
    cache_write_usd: float
    output_usd: float

    @property
    def total_usd(self) -> float:
        """All four classes together."""
        return self.input_usd + self.cache_read_usd + self.cache_write_usd + self.output_usd


def normalize_name(name: str) -> str:
    """The flat key for *name*: trimmed and lowercase."""
    return name.strip().lower()


def candidate_names(model: str) -> tuple[str, ...]:
    """The flat names *model* may be filed under, most specific first.

    A bracketed variant (``[1m]``) is dropped first, since no price differs by
    it. Then come the name as given, the name without a date snapshot suffix,
    and each of those without a ``vendor/`` prefix, so an exact entry wins
    over a looser one.
    """
    name = _BRACKET_SUFFIX.sub("", normalize_name(model))
    bare = name.rpartition("/")[2]
    ordered = (name, _DATE_SUFFIX.sub("", name), bare, _DATE_SUFFIX.sub("", bare))
    return tuple(dict.fromkeys(candidate for candidate in ordered if candidate))


def price_tokens(tokens: TokenUsage, rates: Rates) -> CostBreakdown:
    """What *tokens* cost at *rates*, split by token class.

    Every token is priced once. Reasoning tokens are part of the output count,
    so they are billed at the output rate and not added again.
    """
    write_1h = tokens.cache_write_1h_input_tokens
    write_1h_rate = (
        rates.cache_write_usd if rates.cache_write_1h_usd is None else rates.cache_write_1h_usd
    )
    return CostBreakdown(
        input_usd=tokens.uncached_input_tokens * rates.input_usd / _TOKENS_PER_UNIT,
        cache_read_usd=tokens.cache_read_input_tokens * rates.cache_read_usd / _TOKENS_PER_UNIT,
        cache_write_usd=(
            (tokens.cache_write_input_tokens - write_1h) * rates.cache_write_usd
            + write_1h * write_1h_rate
        )
        / _TOKENS_PER_UNIT,
        output_usd=tokens.output_tokens * rates.output_usd / _TOKENS_PER_UNIT,
    )


def speed_premium_usd(tokens: TokenUsage, entry: RateEntry, speed: UsageSpeed) -> float:
    """How much more *tokens* cost at *speed* than at the standard tier.

    Zero at the standard tier, and zero when the entry lists no rates for
    *speed*: no premium is known, so none is claimed.
    """
    rates = entry.at(speed)
    if rates is None or speed is UsageSpeed.STANDARD:
        return 0.0
    return price_tokens(tokens, rates).total_usd - price_tokens(tokens, entry.standard).total_usd


def cache_savings_usd(tokens: TokenUsage, rates: Rates) -> float:
    """What cache reads saved against paying the input rate for the same tokens."""
    return (
        tokens.cache_read_input_tokens * (rates.input_usd - rates.cache_read_usd) / _TOKENS_PER_UNIT
    )


def _check(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        msg = f"{name} must be a number, got {value!r}"
        raise TypeError(msg)
    if not math.isfinite(value) or value < 0:
        msg = f"{name} must be finite and non-negative, got {value!r}"
        raise ValueError(msg)
