"""Compose provider-owned model facts through the selected catalog."""

from __future__ import annotations

import datetime as dt
import re
from typing import TYPE_CHECKING

from agentshim.core.models import ModelSpec, ModelTable

if TYPE_CHECKING:
    from agentshim.providers.catalog import ProviderCatalog

_DATE_SUFFIX = re.compile(r"-\d{8}$")


def _catalog_or_default(catalog: ProviderCatalog | None) -> ProviderCatalog:
    if catalog is not None:
        return catalog
    from agentshim.providers import DEFAULT_CATALOG  # noqa: PLC0415

    return DEFAULT_CATALOG


def model_spec(
    provider: str,
    model: str | None,
    table: ModelTable | None = None,
    *,
    catalog: ProviderCatalog | None = None,
) -> ModelSpec | None:
    """The facts about *model* run through *provider*, or ``None`` when they are not known.

    Accepts a provider name (``codex``, ``claude``) or a vendor name, and a
    model with a ``vendor/`` prefix (opencode's ``provider/model``). Drops a
    trailing date snapshot (``claude-haiku-4-5-20251001``). Returns ``None``
    for a bracketed variant (``claude-opus-5-5[1m]``), for a provider that
    runs no known vendor's models, and for any model the table lacks; it
    never matches by prefix or family.
    """
    if not model:
        return None
    name = model.strip().lower()
    if "[" in name:
        return None
    source = table if table is not None else default_models(catalog=catalog)
    catalog = _catalog_or_default(catalog)
    provider_name = provider.strip().lower()
    if provider_name in catalog.names():
        resolved = catalog.plugin(provider_name).model_info(
            name, source if table is not None else None
        )
        if resolved is not None or "/" not in name:
            return resolved
    vendors = {entry.vendor for entry in source.entries}
    vendor = provider_name if provider_name in vendors else None
    if "/" in name:
        prefix, _, rest = name.partition("/")
        vendor = prefix if prefix in vendors else None
        name = rest
    if vendor is None or not name:
        return None
    return source.get(vendor, _DATE_SUFFIX.sub("", name))


def context_window(
    provider: str,
    model: str | None,
    table: ModelTable | None = None,
    *,
    catalog: ProviderCatalog | None = None,
) -> int | None:
    """*model*'s context window in tokens when run through *provider*, or ``None`` when not known."""
    spec = model_spec(provider, model, table, catalog=catalog)
    return None if spec is None else spec.context_window


def default_models(*, catalog: ProviderCatalog | None = None) -> ModelTable:
    """Compose model facts supplied by the selected provider catalog."""
    catalog = _catalog_or_default(catalog)
    by_key: dict[tuple[str, str], ModelSpec] = {}
    for name in catalog.names():
        for entry in catalog.plugin(name).known_models():
            previous = by_key.get(entry.key)
            if previous is not None and previous != entry:
                msg = f"conflicting model facts for {entry.vendor}/{entry.model}"
                raise ValueError(msg)
            by_key[entry.key] = entry
    entries = tuple(by_key.values())
    updated = max((entry.checked for entry in entries), default=dt.date(2026, 10, 17))
    return ModelTable(version="2026-10-18.1", last_updated=updated, entries=entries)
