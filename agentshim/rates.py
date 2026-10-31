"""Read flat per-model rates from LiteLLM's price document, and keep them fresh.

This module estimates what tokens would cost at today's list prices. It does
not touch recorded cost: that comes from the pinned table in
:mod:`agentshim.pricing`, which these rates never reach (see
:mod:`agentshim.core.rates`).

agentshim holds no HTTP client and chooses no directory. The caller supplies a
:class:`RateSource` that fetches and a :class:`RateStore` that persists, and
passes the time in, so a refresh is a plain function of its arguments:

* :func:`read_rates` never touches the network. It returns what the store
  holds, or the pinned table shipped in the wheel.
* :func:`refresh_rates` fetches only when the document is older than
  :data:`RATES_TTL` (or ``force`` is set) and no attempt, failed or not, was
  made within :data:`REFRESH_FLOOR`.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

from agentshim.core.errors import AgentShimError
from agentshim.core.plugin import RateCatalog, RateNaming
from agentshim.core.rates import (
    RateDocumentError,
    RateEntry,
    Rates,
    RateTable,
    candidate_names,
    normalize_name,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from agentshim.providers.catalog import ProviderCatalog

__all__ = [
    "LITELLM_ENDPOINTS",
    "RATES_TTL",
    "REFRESH_FLOOR",
    "RateDocumentError",
    "RateEndpoint",
    "RateFetchError",
    "RateRead",
    "RateSnapshot",
    "RateSource",
    "RateStatus",
    "RateStore",
    "bundled_rates",
    "parse_litellm",
    "rates_for",
    "read_rates",
    "refresh_rates",
]

#: How long a fetched document is current before a refresh is due.
RATES_TTL = dt.timedelta(hours=24)
#: The least time between two fetch attempts, after a failure as after a success.
REFRESH_FLOOR = dt.timedelta(seconds=60)

_TOKENS_PER_UNIT = 1_000_000
#: LiteLLM's rates are per token with a few significant digits; rounding to nine decimals of a
#: dollar per million tokens removes the binary noise of the multiplication and nothing else.
_DECIMALS = 9
_FAST_MULTIPLE = "provider_specific_entry"


class RateFetchError(AgentShimError):
    """A :class:`RateSource` could not fetch a document."""


class RateStatus(str, Enum):
    """Where the rates in a :class:`RateRead` came from."""

    #: Fetched within :data:`RATES_TTL`, and no later attempt failed.
    FRESH = "fresh"
    #: Older than the TTL, or the latest refresh failed: the last good document.
    CACHED = "cached"
    #: No usable document: the pinned table shipped with agentshim.
    BUNDLED = "bundled"


@dataclass(frozen=True)
class RateEndpoint:
    """One place to fetch the document from.

    Attributes:
        url: The address.
        accept: The ``Accept`` header to send, since GitHub's contents API
            returns the raw file only when asked for ``application/vnd.github.raw+json``.
    """

    url: str
    accept: str = "application/json"


#: Where LiteLLM publishes its price file: the raw file, then GitHub's API for the same file.
LITELLM_ENDPOINTS = (
    RateEndpoint(
        "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
    ),
    RateEndpoint(
        "https://api.github.com/repos/BerriAI/litellm/contents/model_prices_and_context_window.json"
        "?ref=main",
        accept="application/vnd.github.raw+json",
    ),
)


class RateSource(Protocol):
    """Fetches a rate document. The caller supplies the HTTP client."""

    def fetch(self, endpoint: RateEndpoint) -> bytes:
        """The body at *endpoint*.

        Raises:
            RateFetchError: The request failed or timed out. Any other
                exception is a bug in the source and propagates.
        """
        ...


@dataclass(frozen=True)
class RateSnapshot:
    """What a :class:`RateStore` keeps between runs.

    Attributes:
        document: The last document that parsed, as fetched; ``None`` before the first success.
        fetched_at: When ``document`` was fetched; set exactly when ``document`` is.
        last_attempt_at: When a fetch was last tried, whether it worked or not.
    """

    document: bytes | None
    fetched_at: dt.datetime | None
    last_attempt_at: dt.datetime | None

    def __post_init__(self) -> None:
        """Check the times are timezone-aware and tell one consistent story."""
        if (self.document is None) != (self.fetched_at is None):
            msg = "a snapshot has a fetch time exactly when it has a document"
            raise ValueError(msg)
        for name in ("fetched_at", "last_attempt_at"):
            value: dt.datetime | None = getattr(self, name)
            if value is not None and value.tzinfo is None:
                msg = f"{name} must be timezone-aware"
                raise ValueError(msg)
        if (
            self.fetched_at is not None
            and self.last_attempt_at is not None
            and self.last_attempt_at < self.fetched_at
        ):
            msg = "last_attempt_at must not precede the fetch it follows"
            raise ValueError(msg)


@runtime_checkable
class RateStore(Protocol):
    """Persists one :class:`RateSnapshot`. The caller chooses where."""

    def read(self) -> RateSnapshot | None:
        """The stored snapshot, or ``None`` when nothing was stored."""
        ...

    def write(self, snapshot: RateSnapshot) -> None:
        """Replace the stored snapshot."""
        ...


@dataclass(frozen=True)
class RateRead:
    """Rates and how current they are.

    Attributes:
        table: The rates to price with.
        status: Where they came from.
        fetched_at: When the document behind ``table`` was fetched; ``None`` for the pinned table.
        known_models: How many names ``table`` knows.
        refresh_due: Whether :func:`refresh_rates` would try to fetch now.
    """

    table: RateTable
    status: RateStatus
    fetched_at: dt.datetime | None
    known_models: int
    refresh_due: bool


def parse_litellm(document: bytes, *, checked_on: dt.date) -> RateTable:
    """Read LiteLLM's ``model_prices_and_context_window.json`` into a :class:`RateTable`.

    A model needs finite, non-negative input and output rates, or it is
    dropped; so is one with a malformed value, since a half-read price would
    understate cost. A missing cache read or write rate defaults to the input
    rate. Speed tiers come from ``provider_specific_entry.fast`` (a multiple of
    standard) or from the ``_priority`` (fast) and ``_ultrafast`` columns; a
    tier's missing cache rates scale by the standard ratio. Long-context,
    batch and flex columns are not read.

    Names are trimmed and lowercased. A vendor-prefixed name also files its
    bare name, as an alias, unless that name is already a key. Where several
    prefixed names share a bare name, their speed tiers fill each other's gaps
    and the alias is dropped if any two differ.

    Args:
        document: The file's bytes.
        checked_on: The day the document was read.

    Raises:
        RateDocumentError: The bytes are not a JSON object, or no model in it
            has usable rates.
    """
    try:
        parsed: object = json.loads(document)
    except (ValueError, RecursionError) as error:
        msg = f"rate document is not JSON: {error}"
        raise RateDocumentError(msg) from error
    if not isinstance(parsed, dict):
        msg = "rate document must be a JSON object keyed by model name"
        raise RateDocumentError(msg)
    entries = _read_entries(cast("dict[str, object]", parsed))
    if not entries:
        msg = "rate document has no model with input and output rates"
        raise RateDocumentError(msg)
    return RateTable(checked_on, _with_aliases(entries))


def bundled_rates(*, catalog: ProviderCatalog | None = None) -> RateTable:
    """The pinned table shipped in the wheel: every model a provider prices, with speed tiers.

    It comes from the same generated data as the recorded-cost table, so a
    standard rate cannot differ between the two.
    """
    catalog = _catalog_or_default(catalog)
    entries: dict[str, RateEntry] = {}
    checked: list[dt.date] = []
    for name in catalog.names():
        plugin = catalog.plugin(name)
        if not isinstance(plugin, RateCatalog):
            continue
        known = plugin.known_rates()
        entries.update(known.entries)
        checked.append(known.checked_on)
    return RateTable(max(checked, default=dt.date.min), entries)


def read_rates(
    store: RateStore, *, now: dt.datetime, catalog: ProviderCatalog | None = None
) -> RateRead:
    """The best rates available without the network.

    The stored document when it parses, otherwise the pinned table. Never
    fetches, so it cannot block on a dead network.
    """
    return _read(store.read(), now=now, catalog=catalog)


def refresh_rates(
    store: RateStore,
    source: RateSource,
    *,
    now: dt.datetime,
    force: bool = False,
    catalog: ProviderCatalog | None = None,
) -> RateRead:
    """Fetch a new document if one is due, and report the rates to use.

    A fetch is due when the document is older than :data:`RATES_TTL` (or
    *force*), unless any attempt was made within :data:`REFRESH_FLOOR`. Each
    of :data:`LITELLM_ENDPOINTS` is tried in turn. A document that does not
    parse, or parses to no model, counts as a failed fetch and never replaces
    a good one. The attempt is recorded either way, so the floor holds after a
    failure too.

    Raises:
        Exception: Whatever *store* raises when it cannot write.
    """
    snapshot = store.read()
    if not _due(snapshot, now=now, force=force):
        return _read(snapshot, now=now, catalog=catalog)
    document = _fetch(source, now=now)
    previous = snapshot or RateSnapshot(None, None, None)
    attempted = max(now, previous.fetched_at or now)
    updated = (
        RateSnapshot(document, now, now)
        if document is not None
        else RateSnapshot(previous.document, previous.fetched_at, attempted)
    )
    store.write(updated)
    return _read(updated, now=now, catalog=catalog)


def rates_for(
    table: RateTable, provider: str, model: str, *, catalog: ProviderCatalog | None = None
) -> RateEntry | None:
    """The entry for *model* run through *provider*, or ``None`` when it has no rates.

    The provider decides how its model names map onto the flat names in the
    table: a bracketed variant and a date snapshot are dropped, a ``vendor/``
    prefix is tried last, and a name that is not a model at all (a synthetic
    placeholder, a bare family name) has none. A *provider* the catalog does
    not know gets the generic rules. Never falls back to another model's rates.
    """
    catalog = _catalog_or_default(catalog)
    plugin = (
        catalog.plugin(provider.strip().lower())
        if provider.strip().lower() in catalog.names()
        else None
    )
    names = plugin.rate_names(model) if isinstance(plugin, RateNaming) else candidate_names(model)
    for name in names:
        entry = table.lookup(name)
        if entry is not None:
            return entry
    return None


def _catalog_or_default(catalog: ProviderCatalog | None) -> ProviderCatalog:
    if catalog is not None:
        return catalog
    from agentshim.providers import DEFAULT_CATALOG  # noqa: PLC0415

    return DEFAULT_CATALOG


def _read(
    snapshot: RateSnapshot | None, *, now: dt.datetime, catalog: ProviderCatalog | None
) -> RateRead:
    due = _due(snapshot, now=now, force=False)
    table = _parse_stored(snapshot)
    if snapshot is None or snapshot.fetched_at is None or table is None:
        bundled = bundled_rates(catalog=catalog)
        return RateRead(bundled, RateStatus.BUNDLED, None, len(bundled), due)
    failed_since = (
        snapshot.last_attempt_at is not None and snapshot.last_attempt_at > snapshot.fetched_at
    )
    fresh = now - snapshot.fetched_at < RATES_TTL and not failed_since
    status = RateStatus.FRESH if fresh else RateStatus.CACHED
    return RateRead(table, status, snapshot.fetched_at, len(table), due)


def _parse_stored(snapshot: RateSnapshot | None) -> RateTable | None:
    if snapshot is None or snapshot.document is None or snapshot.fetched_at is None:
        return None
    day = snapshot.fetched_at.astimezone(dt.timezone.utc).date()
    try:
        return parse_litellm(snapshot.document, checked_on=day)
    except RateDocumentError:
        return None


def _due(snapshot: RateSnapshot | None, *, now: dt.datetime, force: bool) -> bool:
    if snapshot is None:
        return True
    if snapshot.last_attempt_at is not None and now - snapshot.last_attempt_at < REFRESH_FLOOR:
        return False
    if force or snapshot.fetched_at is None:
        return True
    return now - snapshot.fetched_at >= RATES_TTL


def _fetch(source: RateSource, *, now: dt.datetime) -> bytes | None:
    for endpoint in LITELLM_ENDPOINTS:
        try:
            document = source.fetch(endpoint)
            parse_litellm(document, checked_on=now.astimezone(dt.timezone.utc).date())
        except (RateFetchError, RateDocumentError):
            continue
        return document
    return None


def _read_entries(document: Mapping[str, object]) -> dict[str, RateEntry]:
    entries: dict[str, RateEntry] = {}
    for name, raw in document.items():
        key = normalize_name(name)
        if not key or not isinstance(raw, dict):
            continue
        fields = cast("Mapping[str, object]", raw)
        standard = _read_rates(fields, "", None)
        if standard is None:
            continue
        entries[key] = RateEntry(
            key,
            standard,
            fast=_read_fast(fields, standard),
            ultrafast=_read_rates(fields, "_ultrafast", standard),
        )
    return entries


def _read_fast(fields: Mapping[str, object], standard: Rates) -> Rates | None:
    specific = fields.get(_FAST_MULTIPLE)
    if isinstance(specific, dict):
        multiple = _finite(cast("Mapping[str, object]", specific).get("fast"))
        if multiple is not None and multiple > 0:
            return _rounded(standard.scaled(multiple))
    return _read_rates(fields, "_priority", standard)


def _read_rates(fields: Mapping[str, object], suffix: str, standard: Rates | None) -> Rates | None:
    """Rates from the columns ending in *suffix*; a tier's gaps scale from *standard*."""
    input_usd = _per_million(fields.get(f"input_cost_per_token{suffix}"))
    output_usd = _per_million(fields.get(f"output_cost_per_token{suffix}"))
    if input_usd is None or output_usd is None:
        return None

    def gap(column: str, held: float | None) -> float | None:
        value = _per_million(fields.get(f"{column}{suffix}"))
        if value is not None:
            return value
        if standard is None or held is None or standard.input_usd <= 0:
            return None
        return round(held / standard.input_usd * input_usd, _DECIMALS)

    return Rates(
        input_usd=input_usd,
        output_usd=output_usd,
        cache_read_usd=_default(
            gap("cache_read_input_token_cost", _held(standard, "read")), input_usd
        ),
        cache_write_usd=_default(
            gap("cache_creation_input_token_cost", _held(standard, "write")), input_usd
        ),
        cache_write_1h_usd=gap(
            "cache_creation_input_token_cost_above_1hr", _held(standard, "write_1h")
        ),
    )


def _held(standard: Rates | None, column: str) -> float | None:
    if standard is None:
        return None
    return {
        "read": standard.cache_read_usd,
        "write": standard.cache_write_usd,
        "write_1h": standard.cache_write_1h_usd,
    }[column]


def _default(value: float | None, fallback: float) -> float:
    return fallback if value is None else value


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) else None


def _per_million(value: object) -> float | None:
    """USD per token to USD per million tokens; ``None`` for a missing, negative or odd value."""
    number = _finite(value)
    if number is None or number < 0:
        return None
    scaled = round(number * _TOKENS_PER_UNIT, _DECIMALS)
    return scaled if math.isfinite(scaled) else None


def _rounded(rates: Rates) -> Rates:
    write_1h = rates.cache_write_1h_usd
    return Rates(
        input_usd=round(rates.input_usd, _DECIMALS),
        output_usd=round(rates.output_usd, _DECIMALS),
        cache_read_usd=round(rates.cache_read_usd, _DECIMALS),
        cache_write_usd=round(rates.cache_write_usd, _DECIMALS),
        cache_write_1h_usd=None if write_1h is None else round(write_1h, _DECIMALS),
    )


def _with_aliases(entries: Mapping[str, RateEntry]) -> dict[str, RateEntry]:
    """Add each vendor-prefixed name's bare name, unless taken or its sources disagree."""
    held: dict[str, RateEntry | None] = {}
    for key, entry in entries.items():
        alias = key.rpartition("/")[2]
        if not alias or alias == key or alias in entries:
            continue
        candidate = RateEntry(alias, entry.standard, entry.fast, entry.ultrafast)
        held[alias] = _merged(held[alias], candidate) if alias in held else candidate
    merged = dict(entries)
    merged.update({alias: entry for alias, entry in held.items() if entry is not None})
    return merged


def _merged(first: RateEntry | None, second: RateEntry) -> RateEntry | None:
    """One entry from two that share a bare name, or ``None`` if they conflict.

    Resellers often omit a speed tier, so a tier one lacks is filled from the
    other; a tier both list must agree.
    """
    if first is None or first.standard != second.standard:
        return None
    fast = first.fast or second.fast
    ultrafast = first.ultrafast or second.ultrafast
    if fast != (second.fast or fast) or ultrafast != (second.ultrafast or ultrafast):
        return None
    return RateEntry(first.name, first.standard, fast, ultrafast)
