"""Anthropic rates, generated from a LiteLLM snapshot. Do not edit.

Regenerate with ``python scripts/generate_prices.py``; ``docs/pricing.md`` says when.
"""

import datetime as dt

from agentshim.core.rates import RateEntry, Rates

SNAPSHOT_DATE = dt.date(2026, 10, 30)
SNAPSHOT_SHA256 = "03b069b3018a604a4dd68cdbdb1d6c5a05a2041887e9447bdcc203f18be9fb36"

#: Models whose cache write rate the snapshot does not list: it is priced at base input.
INFERRED_CACHE_WRITE = frozenset[str]()

RATE_ENTRIES = (
    RateEntry(
        "claude-fable-5-1",
        Rates(
            input_usd=10.0,
            output_usd=50.0,
            cache_read_usd=0.25,
            cache_write_usd=12.5,
            cache_write_1h_usd=20.0,
        ),
    ),
    RateEntry(
        "claude-fable-5",
        Rates(
            input_usd=10.0,
            output_usd=50.0,
            cache_read_usd=1.0,
            cache_write_usd=12.5,
            cache_write_1h_usd=20.0,
        ),
    ),
    RateEntry(
        "claude-opus-5-5",
        Rates(
            input_usd=4.0,
            output_usd=20.0,
            cache_read_usd=0.2,
            cache_write_usd=5.0,
            cache_write_1h_usd=8.0,
        ),
        fast=Rates(
            input_usd=8.0,
            output_usd=40.0,
            cache_read_usd=0.4,
            cache_write_usd=10.0,
            cache_write_1h_usd=16.0,
        ),
    ),
    RateEntry(
        "claude-opus-5",
        Rates(
            input_usd=5.0,
            output_usd=25.0,
            cache_read_usd=0.5,
            cache_write_usd=6.25,
            cache_write_1h_usd=10.0,
        ),
        fast=Rates(
            input_usd=10.0,
            output_usd=50.0,
            cache_read_usd=1.0,
            cache_write_usd=12.5,
            cache_write_1h_usd=20.0,
        ),
    ),
    RateEntry(
        "claude-opus-4-8",
        Rates(
            input_usd=5.0,
            output_usd=25.0,
            cache_read_usd=0.5,
            cache_write_usd=6.25,
            cache_write_1h_usd=10.0,
        ),
        fast=Rates(
            input_usd=10.0,
            output_usd=50.0,
            cache_read_usd=1.0,
            cache_write_usd=12.5,
            cache_write_1h_usd=20.0,
        ),
    ),
    RateEntry(
        "claude-opus-4-7",
        Rates(
            input_usd=5.0,
            output_usd=25.0,
            cache_read_usd=0.5,
            cache_write_usd=6.25,
            cache_write_1h_usd=10.0,
        ),
    ),
    RateEntry(
        "claude-opus-4-6",
        Rates(
            input_usd=5.0,
            output_usd=25.0,
            cache_read_usd=0.5,
            cache_write_usd=6.25,
            cache_write_1h_usd=10.0,
        ),
    ),
    RateEntry(
        "claude-sonnet-5-5",
        Rates(
            input_usd=2.0,
            output_usd=10.0,
            cache_read_usd=0.1,
            cache_write_usd=2.5,
            cache_write_1h_usd=4.0,
        ),
    ),
    RateEntry(
        "claude-sonnet-5",
        Rates(
            input_usd=2.0,
            output_usd=10.0,
            cache_read_usd=0.2,
            cache_write_usd=2.5,
            cache_write_1h_usd=4.0,
        ),
    ),
    RateEntry(
        "claude-sonnet-4-6",
        Rates(
            input_usd=3.0,
            output_usd=15.0,
            cache_read_usd=0.3,
            cache_write_usd=3.75,
            cache_write_1h_usd=6.0,
        ),
    ),
    RateEntry(
        "claude-haiku-5-5",
        Rates(
            input_usd=0.1,
            output_usd=0.5,
            cache_read_usd=0.01,
            cache_write_usd=0.125,
            cache_write_1h_usd=0.2,
        ),
    ),
    RateEntry(
        "claude-haiku-4-5",
        Rates(
            input_usd=1.0,
            output_usd=5.0,
            cache_read_usd=0.1,
            cache_write_usd=1.25,
            cache_write_1h_usd=2.0,
        ),
    ),
)
