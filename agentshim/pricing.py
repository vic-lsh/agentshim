"""Compose provider-owned prices through the selected catalog."""

from __future__ import annotations

import datetime as dt
import re
from typing import TYPE_CHECKING

from agentshim.core.plugin import PricingContext
from agentshim.core.pricing import ModelPricing, PricingTable, cost_usd

__all__ = [
    "ModelPricing",
    "PricingTable",
    "cost_usd",
    "default_pricing",
    "price_for",
    "vendor_and_model",
]

if TYPE_CHECKING:
    from agentshim.providers.catalog import ProviderCatalog

_DATE_SUFFIX = re.compile(r"-\d{8}$")
_BRACKET_SUFFIX = re.compile(r"\[[^\]]*\]$")


def _catalog_or_default(catalog: ProviderCatalog | None) -> ProviderCatalog:
    if catalog is not None:
        return catalog
    from agentshim.providers import DEFAULT_CATALOG  # noqa: PLC0415

    return DEFAULT_CATALOG


def vendor_and_model(
    provider: str,
    model: str,
    table: PricingTable | None = None,
    *,
    catalog: ProviderCatalog | None = None,
) -> tuple[str, str] | None:
    """Resolve an agentshim provider and model name to a table key.

    Accepts a provider name (``codex``, ``claude``) or a vendor name, and a
    model with a ``vendor/`` prefix (opencode's ``provider/model``). Drops a
    trailing date snapshot (``claude-haiku-4-5-20251001``) and a bracketed
    variant (``claude-opus-5-5[1m]``). Returns ``None`` when the provider is
    not billed per token by a known vendor.
    """
    entry = price_for(provider, model, table, catalog=catalog)
    return None if entry is None else entry.key


def price_for(
    provider: str,
    model: str | None,
    table: PricingTable | None = None,
    *,
    catalog: ProviderCatalog | None = None,
    context: PricingContext | None = None,
) -> ModelPricing | None:
    """The price of *model* run through *provider*, or ``None`` when it is not known.

    Never falls back to another model's price or to zero.
    """
    if not model:
        return None
    source = table if table is not None else default_pricing(catalog=catalog)
    catalog = _catalog_or_default(catalog)
    provider_name = provider.strip().lower()
    if provider_name in catalog.names():
        resolved = catalog.plugin(provider_name).pricing_info(
            model, context or PricingContext(), source if table is not None else None
        )
        if resolved is not None or "/" not in model:
            return resolved
    name = model.strip().lower()
    vendors = {entry.vendor for entry in source.entries}
    vendor = provider_name if provider_name in vendors else None
    if "/" in name:
        prefix, _, rest = name.partition("/")
        vendor = prefix if prefix in vendors else None
        name = rest
    if vendor is None or not name:
        return None
    if context is not None and (
        context.as_of is not None
        or context.region not in (None, "global")
        or context.tier not in (None, "standard")
    ):
        return None
    return source.get(vendor, _DATE_SUFFIX.sub("", _BRACKET_SUFFIX.sub("", name)))


def default_pricing(*, catalog: ProviderCatalog | None = None) -> PricingTable:
    """Compose prices supplied by the selected provider catalog."""
    catalog = _catalog_or_default(catalog)
    by_key: dict[tuple[str, str], ModelPricing] = {}
    for name in catalog.names():
        for entry in catalog.plugin(name).known_prices():
            previous = by_key.get(entry.key)
            if previous is not None and previous != entry:
                msg = f"conflicting prices for {entry.vendor}/{entry.model}"
                raise ValueError(msg)
            by_key[entry.key] = entry
    entries = tuple(by_key.values())
    updated = max((entry.checked for entry in entries), default=dt.date(2026, 9, 27))
    return PricingTable(version="2026-10-30", last_updated=updated, entries=entries)
