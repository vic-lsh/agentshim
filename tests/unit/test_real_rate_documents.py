"""Fetched rate documents as LiteLLM publishes them, and a clock that is not steady.

``test_rates.py`` reads small complete documents. These hold the invariants over
``rate_shapes.py``: a document that omits a model, renames it, or lists a placeholder at zero must
never take a known price away. Each test but the last two failed on 0.24.0, which read the fetched
document alone.
"""

from __future__ import annotations

import datetime as dt

import pytest
from agentshim import (
    RateSnapshot,
    RateStatus,
    TokenUsage,
    cost_usd,
    parse_litellm,
    price_for,
    price_tokens,
    rates_for,
    read_rates,
    refresh_rates,
)
from agentshim.rates import LITELLM_ENDPOINTS, REFRESH_FLOOR, RateFetchError
from agentshim.testing import FakeRateSource, FakeRateStore
from hypothesis import given, settings
from hypothesis import strategies as st

from .rate_shapes import NAMES, PINNED, document_bytes, fetched_documents

NOON = dt.datetime(2026, 10, 30, 12, 0, tzinfo=dt.timezone.utc)
DAY = NOON.date()
GOOD = document_bytes({"some-model": {"input_cost_per_token": 1e-6, "output_cost_per_token": 2e-6}})


def _priced(rates: object) -> bool:
    return rates.input_usd > 0 and rates.output_usd > 0  # type: ignore[attr-defined]


class TestAFetchedDocumentNeverTakesAPriceAway:
    @settings(max_examples=120)
    @given(shape=fetched_documents())
    def test_every_pinned_model_stays_priced_and_the_document_wins_only_where_it_prices(
        self, shape: tuple[dict, dict[str, str]]
    ) -> None:
        document, fates = shape
        payload = document_bytes(document)
        read = read_rates(FakeRateStore(RateSnapshot(payload, NOON, NOON)), now=NOON)
        try:
            parsed = parse_litellm(payload, checked_on=DAY)
        except Exception:  # noqa: BLE001 - a document with no usable model is read as none at all
            assert read.status is RateStatus.BUNDLED
            parsed = None
        for name in NAMES:
            entry = read.table.lookup(name)
            assert entry is not None, f"{name} lost its rates ({fates[name]})"
            assert _priced(entry.standard), f"{name} priced at zero ({fates[name]})"
            for tier in (entry.fast, entry.ultrafast):
                assert tier is None or _priced(tier), f"{name} has a free speed tier"
            if fates[name] in ("same", "repriced", "zero_fast") and parsed is not None:
                assert entry.standard == parsed.entries[name].standard
            else:
                assert entry.standard == PINNED.entries[name].standard
        assert len(read.table) >= len(PINNED)

    def test_a_model_the_document_dropped_keeps_its_pinned_rates(self) -> None:
        read = read_rates(FakeRateStore(RateSnapshot(GOOD, NOON, NOON)), now=NOON)
        assert read.status is RateStatus.FRESH
        assert rates_for(read.table, "claude", "claude-opus-5-5") is not None
        assert rates_for(read.table, "claude", "claude-opus-5-5") == PINNED.lookup(
            "claude-opus-5-5"
        )
        assert read.table.lookup("some-model") is not None

    def test_a_placeholder_row_at_zero_does_not_replace_a_pinned_price(self) -> None:
        zero = document_bytes(
            {"claude-opus-5-5": {"input_cost_per_token": 0, "output_cost_per_token": 0}}
        )
        read = read_rates(FakeRateStore(RateSnapshot(zero, NOON, NOON)), now=NOON)
        entry = rates_for(read.table, "claude", "claude-opus-5-5")
        assert entry is not None
        assert entry.standard.input_usd > 0
        # The same row for a model agentshim has no price for is taken as listed.
        free = document_bytes(
            {"some-free-model": {"input_cost_per_token": 0, "output_cost_per_token": 0}}
        )
        table = read_rates(FakeRateStore(RateSnapshot(free, NOON, NOON)), now=NOON).table
        assert table.lookup("some-free-model") is not None

    def test_a_refresh_lays_what_it_fetched_over_the_pinned_table_too(self) -> None:
        store = FakeRateStore()
        source = FakeRateSource(GOOD)
        read = refresh_rates(store, source, now=NOON)
        assert read.status is RateStatus.FRESH
        assert read.table.lookup("claude-opus-5-5") is not None
        assert read.known_models == len(PINNED) + 1

    @settings(max_examples=60)
    @given(
        tokens=st.builds(
            TokenUsage,
            input_tokens=st.integers(min_value=0, max_value=10**7),
            output_tokens=st.integers(min_value=0, max_value=10**7),
        ),
        name=st.sampled_from(NAMES),
    )
    def test_the_estimate_at_the_standard_tier_is_the_recorded_cost(
        self, tokens: TokenUsage, name: str
    ) -> None:
        # Holds on 0.24.0 too. Nothing asserted it: the two prices come from one generated table.
        entry = PINNED.entries[name]
        vendor = "codex" if name.startswith(("gpt", "o")) else "claude"
        pricing = price_for(vendor, name)
        assert pricing is not None
        recorded = cost_usd(tokens, pricing)
        assert price_tokens(tokens, entry.standard).total_usd == pytest.approx(recorded, rel=1e-9)


class TestAnUnsteadyClock:
    def test_a_naive_time_is_refused_before_anything_is_fetched(self) -> None:
        naive = NOON.replace(tzinfo=None)
        source = FakeRateSource(GOOD)
        with pytest.raises(ValueError, match="now"):
            refresh_rates(FakeRateStore(), source, now=naive)
        assert source.requests == []
        with pytest.raises(ValueError, match="now"):
            read_rates(FakeRateStore(), now=naive)

    @settings(max_examples=80)
    @given(back=st.integers(min_value=int(REFRESH_FLOOR.total_seconds()) + 1, max_value=10**6))
    def test_a_clock_set_back_does_not_hold_off_refreshing_until_it_catches_up(
        self, back: int
    ) -> None:
        store = FakeRateStore()
        failing = FakeRateSource(RateFetchError("down"), RateFetchError("down"))
        refresh_rates(store, failing, now=NOON)
        within = FakeRateSource(GOOD)
        refresh_rates(store, within, now=NOON - dt.timedelta(seconds=30), force=True)
        assert within.requests == []  # an attempt within the floor either side holds the floor
        earlier = NOON - dt.timedelta(seconds=back)
        source = FakeRateSource(GOOD)
        read = refresh_rates(store, source, now=earlier, force=True)
        assert source.requests[:1] == [LITELLM_ENDPOINTS[0]]
        assert read.status is RateStatus.FRESH

    @settings(max_examples=80)
    @given(back=st.integers(min_value=int(REFRESH_FLOOR.total_seconds()) + 1, max_value=10**6))
    def test_a_failed_fetch_after_the_clock_was_set_back_records_the_attempt_when_it_was_made(
        self, back: int
    ) -> None:
        store = FakeRateStore(RateSnapshot(GOOD, NOON, NOON))
        earlier = NOON - dt.timedelta(seconds=back)
        down = FakeRateSource(*[RateFetchError("down")] * len(LITELLM_ENDPOINTS))
        read = refresh_rates(store, down, now=earlier)
        assert store.snapshot is not None
        assert store.snapshot.last_attempt_at == earlier
        assert read.status is RateStatus.CACHED
        again = FakeRateSource(GOOD)
        refresh_rates(store, again, now=earlier + dt.timedelta(seconds=1))
        assert again.requests == []  # the floor holds from the attempt that failed

    @settings(max_examples=80)
    @given(back=st.integers(min_value=int(REFRESH_FLOOR.total_seconds()) + 1, max_value=10**6))
    def test_a_document_stamped_by_a_clock_that_ran_ahead_is_not_fresh_forever(
        self, back: int
    ) -> None:
        store = FakeRateStore(RateSnapshot(GOOD, NOON, NOON))
        read = read_rates(store, now=NOON - dt.timedelta(seconds=back))
        assert read.status is RateStatus.CACHED
        assert read.refresh_due
