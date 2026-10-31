"""Generate the pinned price tables from a checked-in LiteLLM snapshot.

LiteLLM's ``model_prices_and_context_window.json`` is the one machine-readable
source of model prices. A release reads it once into
``tests/fixtures/litellm/snapshot.json`` (trimmed to the models agentshim
prices, with the day it was read and the SHA-256 of the whole file) and
generates ``prices_generated.py`` for each priced provider from that snapshot,
using the library's own :func:`agentshim.rates.parse_litellm`. A run never
touches the network: it uses the generated tables.

Three steps, each usable on its own:

1. ``--fetch`` (or ``--from-file FILE``) reads the full LiteLLM document, keeps
   the models in :data:`VENDORS`, and writes the snapshot. Only this step
   touches the network, and only a release runs it.
2. Without those flags the checked-in snapshot is the input, so the output is
   reproducible offline.
3. The result is written to each provider's ``prices_generated.py``;
   ``--check`` compares instead of writing and exits non-zero on drift.

A model in :data:`VENDORS` that the snapshot lacks stops the release: add the
model to the snapshot's source, or take it out of the list and, if the
catalogue lists it, name it with its reason in ``UNPRICED_BY_DESIGN`` in
``tests/unit/test_pricing.py``. A model not listed here is never priced, so a
future LiteLLM row for an excluded model is not priced silently.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agentshim.rates import LITELLM_ENDPOINTS, parse_litellm

if TYPE_CHECKING:
    from agentshim.core.rates import RateEntry, Rates

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_PATH = ROOT / "tests" / "fixtures" / "litellm" / "snapshot.json"
FETCH_TIMEOUT_S = 60

#: The columns the parser reads; the snapshot keeps these and drops the rest of LiteLLM's row.
_KEPT_COLUMN = re.compile(
    r"^(input_cost_per_token|output_cost_per_token|cache_read_input_token_cost"
    r"|cache_creation_input_token_cost(_above_1hr)?)(_priority|_ultrafast)?$"
)
_KEPT_FIELDS = frozenset({"provider_specific_entry", "litellm_provider", "source"})


@dataclass(frozen=True)
class Vendor:
    """One vendor's priced models and where they go.

    Attributes:
        vendor: The billing vendor, as ``ModelPricing.vendor`` names it.
        package: The provider package that owns the output.
        models: LiteLLM keys to price, in table order.
    """

    vendor: str
    package: str
    models: tuple[str, ...]

    @property
    def output_path(self) -> Path:
        """Where this vendor's generated module is written."""
        return ROOT / "agentshim" / "providers" / self.package / "prices_generated.py"


VENDORS = (
    Vendor(
        "anthropic",
        "claude",
        (
            "claude-fable-5-1",
            "claude-fable-5",
            "claude-opus-5-5",
            "claude-opus-5",
            "claude-opus-4-8",
            "claude-opus-4-7",
            "claude-opus-4-6",
            "claude-sonnet-5-5",
            "claude-sonnet-5",
            "claude-sonnet-4-6",
            "claude-haiku-5-5",
            "claude-haiku-4-5",
        ),
    ),
    Vendor(
        "openai",
        "codex",
        (
            "gpt-6-astra",
            "gpt-6.1-sol",
            "gpt-6-sol",
            "gpt-6-luna",
            "gpt-5.6-sol",
            "gpt-5.6-terra",
            "gpt-5.6-luna",
            "gpt-5.5",
            "gpt-5.4",
            "gpt-5.4-mini",
            "gpt-5.4-nano",
            "gpt-5.3-codex",
            "gpt-5.2",
            "gpt-5.1",
            "gpt-5",
            "gpt-5-mini",
            "gpt-5-nano",
        ),
    ),
)


class GeneratorError(Exception):
    """The snapshot or its configuration cannot produce a table."""


def trim(document: dict[str, Any]) -> dict[str, Any]:
    """The configured models of a full LiteLLM document, with only the columns the parser reads.

    Raises:
        GeneratorError: A configured model is missing from *document*.
    """
    wanted = [model for vendor in VENDORS for model in vendor.models]
    missing = [model for model in wanted if not isinstance(document.get(model), dict)]
    if missing:
        msg = f"LiteLLM has no row for {', '.join(missing)}"
        raise GeneratorError(msg)
    return {
        model: {
            key: value
            for key, value in document[model].items()
            if key in _KEPT_FIELDS or _KEPT_COLUMN.match(key)
        }
        for model in wanted
    }


def make_snapshot(raw: bytes, *, fetched_on: dt.date, source: str) -> dict[str, Any]:
    """The snapshot for a full LiteLLM document read on *fetched_on* from *source*."""
    document = json.loads(raw)
    if not isinstance(document, dict):
        msg = "the LiteLLM document is not a JSON object"
        raise GeneratorError(msg)
    return {
        "fetched_on": fetched_on.isoformat(),
        "source": source,
        "upstream_sha256": hashlib.sha256(raw).hexdigest(),
        "upstream_bytes": len(raw),
        "models": trim(document),
    }


def read_snapshot(path: Path = SNAPSHOT_PATH) -> dict[str, Any]:
    """The checked-in snapshot."""
    return json.loads(path.read_text())


def render_snapshot(snapshot: dict[str, Any]) -> str:
    """The snapshot file's text: stable key order, one level of indent."""
    return json.dumps(snapshot, indent=1, sort_keys=True) + "\n"


def _rates(rates: Rates, indent: str) -> str:
    fields = [
        f"input_usd={rates.input_usd!r}",
        f"output_usd={rates.output_usd!r}",
        f"cache_read_usd={rates.cache_read_usd!r}",
        f"cache_write_usd={rates.cache_write_usd!r}",
    ]
    if rates.cache_write_1h_usd is not None:
        fields.append(f"cache_write_1h_usd={rates.cache_write_1h_usd!r}")
    body = "".join(f"{indent}    {field},\n" for field in fields)
    return f"Rates(\n{body}{indent})"


def _entry(entry: RateEntry) -> str:
    lines = [
        f"    RateEntry(\n        {json.dumps(entry.name)},\n        {_rates(entry.standard, '        ')},"
    ]
    for tier in ("fast", "ultrafast"):
        rates: Rates | None = getattr(entry, tier)
        if rates is not None:
            lines.append(f"        {tier}={_rates(rates, '        ')},")
    lines.append("    ),")
    return "\n".join(lines)


def render(vendor: Vendor, snapshot: dict[str, Any]) -> str:
    """The generated module for *vendor*, from *snapshot*."""
    fetched_on = dt.date.fromisoformat(snapshot["fetched_on"])
    models = {model: snapshot["models"][model] for model in vendor.models}
    table = parse_litellm(json.dumps(models).encode(), checked_on=fetched_on)
    entries = [table.entries[model] for model in vendor.models]
    inferred = [
        model for model in vendor.models if "cache_creation_input_token_cost" not in models[model]
    ]
    inferred_text = (
        "frozenset(\n    {\n"
        + "".join(f"        {json.dumps(name)},\n" for name in inferred)
        + "    }\n)"
        if inferred
        else "frozenset[str]()"
    )
    return (
        f'"""{vendor.vendor.capitalize()} rates, generated from a LiteLLM snapshot. Do not edit.\n'
        "\n"
        "Regenerate with ``python scripts/generate_prices.py``; ``docs/pricing.md`` says when.\n"
        '"""\n'
        "\n"
        "import datetime as dt\n"
        "\n"
        "from agentshim.core.rates import RateEntry, Rates\n"
        "\n"
        f"SNAPSHOT_DATE = dt.date({fetched_on.year}, {fetched_on.month}, {fetched_on.day})\n"
        f"SNAPSHOT_SHA256 = {json.dumps(snapshot['upstream_sha256'])}\n"
        "\n"
        "#: Models whose cache write rate the snapshot does not list: it is priced at base input.\n"
        f"INFERRED_CACHE_WRITE = {inferred_text}\n"
        "\n"
        "RATE_ENTRIES = (\n" + "\n".join(_entry(entry) for entry in entries) + "\n)\n"
    )


def outputs(snapshot: dict[str, Any]) -> dict[Path, str]:
    """Every file the snapshot generates, by path."""
    return {vendor.output_path: render(vendor, snapshot) for vendor in VENDORS}


def _fetch() -> tuple[bytes, str]:
    last: Exception | None = None
    for endpoint in LITELLM_ENDPOINTS:
        request = urllib.request.Request(  # noqa: S310 - https URLs from LITELLM_ENDPOINTS
            endpoint.url, headers={"Accept": endpoint.accept}
        )
        try:
            with urllib.request.urlopen(request, timeout=FETCH_TIMEOUT_S) as response:  # noqa: S310
                return response.read(), endpoint.url
        except OSError as error:
            last = error
    msg = f"could not fetch LiteLLM: {last}"
    raise GeneratorError(msg)


def main(argv: list[str] | None = None) -> int:
    """Write the snapshot and tables, or with ``--check`` report whether they differ."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare instead of writing")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--fetch", action="store_true", help="read today's LiteLLM document")
    source.add_argument("--from-file", type=Path, help="read a LiteLLM document saved on disk")
    parser.add_argument("--date", type=dt.date.fromisoformat, help="the day the document was read")
    args = parser.parse_args(argv)

    if args.fetch or args.from_file:
        raw, origin = (
            _fetch() if args.fetch else (args.from_file.read_bytes(), LITELLM_ENDPOINTS[0].url)
        )
        fetched_on = args.date or dt.datetime.now(dt.timezone.utc).date()
        snapshot = make_snapshot(raw, fetched_on=fetched_on, source=origin)
    else:
        snapshot = read_snapshot()

    files = {SNAPSHOT_PATH: render_snapshot(snapshot), **outputs(snapshot)}
    if args.check:
        stale = [path for path, text in files.items() if path.read_text() != text]
        for path in stale:
            print(
                f"{path.relative_to(ROOT)} differs from what the snapshot generates",
                file=sys.stderr,
            )
        return 1 if stale else 0
    for path, text in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
