"""Flat rates: the LiteLLM parser, the price arithmetic, and the refresh policy."""

from __future__ import annotations

import contextlib
import datetime as dt
import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from agentshim import (
    CostBreakdown,
    ModelPricing,
    RateDocumentError,
    RateEntry,
    Rates,
    RateSnapshot,
    RateStatus,
    RateTable,
    TokenUsage,
    UsageSpeed,
    cache_savings_usd,
    cost_usd,
    default_pricing,
    parse_litellm,
    price_tokens,
    read_rates,
    refresh_rates,
    speed_premium_usd,
)
from agentshim.rates import (
    LITELLM_ENDPOINTS,
    RATES_TTL,
    REFRESH_FLOOR,
    RateFetchError,
    bundled_rates,
)
from agentshim.testing import FakeRateSource, FakeRateStore
from agentshim.testing.contracts import RateStoreContract
from hypothesis import given
from hypothesis import strategies as st

if TYPE_CHECKING:
    from collections.abc import Callable

TRIMMED = Path(__file__).resolve().parents[1] / "fixtures" / "litellm" / "trimmed.json"
DAY = dt.date(2026, 10, 30)
NOON = dt.datetime(2026, 10, 30, 12, 0, tzinfo=dt.timezone.utc)


def _parse(document: object) -> RateTable:
    return parse_litellm(json.dumps(document).encode(), checked_on=DAY)


@pytest.fixture(scope="module")
def trimmed() -> RateTable:
    return parse_litellm(TRIMMED.read_bytes(), checked_on=DAY)


class TestParser:
    def test_a_standard_row_with_a_fast_multiple(self, trimmed: RateTable) -> None:
        entry = trimmed.lookup("alpha")
        assert entry is not None
        assert entry.standard == Rates(1.0, 4.0, 0.1, 1.25, 2.0)
        assert entry.fast == Rates(2.0, 8.0, 0.2, 2.5, 4.0)
        assert entry.ultrafast is None

    def test_missing_cache_rates_default_to_the_input_rate(self, trimmed: RateTable) -> None:
        entry = trimmed.lookup("bare-cache")
        assert entry is not None
        assert entry.standard == Rates(3.0, 6.0, 3.0, 3.0, None)

    def test_a_priority_tier_scales_missing_cache_rates_by_the_standard_ratio(
        self, trimmed: RateTable
    ) -> None:
        entry = trimmed.lookup("priority-tier")
        assert entry is not None
        assert entry.standard == Rates(2.0, 8.0, 0.2, 2.0)
        assert entry.fast == Rates(4.0, 16.0, 0.4, 4.0)

    def test_an_ultrafast_tier_keeps_a_cache_rate_it_lists_and_scales_the_rest(
        self, trimmed: RateTable
    ) -> None:
        entry = trimmed.lookup("ultra-tier")
        assert entry is not None
        assert entry.ultrafast == Rates(12.0, 48.0, 0.6, 15.0)
        assert entry.fast is None

    def test_a_free_model_is_kept(self, trimmed: RateTable) -> None:
        entry = trimmed.lookup("free")
        assert entry is not None
        assert entry.standard == Rates(0.0, 0.0, 0.0, 0.0)

    @pytest.mark.parametrize(
        "name", ["sample_spec", "no-output", "negative", "string-rate", "boolean-rate"]
    )
    def test_a_row_without_two_valid_rates_is_dropped(self, trimmed: RateTable, name: str) -> None:
        assert trimmed.lookup(name) is None

    def test_names_are_trimmed_and_lowercased(self, trimmed: RateTable) -> None:
        assert trimmed.lookup("MIXED-case") is not None
        assert "  Mixed-Case  " not in trimmed.entries

    def test_agreeing_prefixed_names_file_a_bare_alias_and_fill_a_missing_tier(
        self, trimmed: RateTable
    ) -> None:
        alias = trimmed.lookup("shared")
        assert alias is not None
        assert alias.standard == Rates(1.0, 2.0, 1.0, 1.0)
        assert alias.fast == Rates(3.0, 6.0, 3.0, 3.0)
        assert trimmed.lookup("vendor-a/shared") is not None

    @pytest.mark.parametrize("name", ["conflict", "fastclash"])
    def test_a_bare_alias_whose_sources_differ_is_dropped(
        self, trimmed: RateTable, name: str
    ) -> None:
        assert trimmed.lookup(name) is None
        assert trimmed.lookup(f"vendor-a/{name}") is not None

    def test_an_alias_never_replaces_a_name_the_document_has(self, trimmed: RateTable) -> None:
        entry = trimmed.lookup("alpha")
        assert entry is not None
        assert entry.standard.input_usd == 1.0

    @pytest.mark.parametrize(
        "document", [b"", b"not json", b"[]", b'"x"', b"{}", b'{"m": 1}', b'{"m": {}}']
    )
    def test_a_document_without_a_usable_model_is_refused(self, document: bytes) -> None:
        with pytest.raises(RateDocumentError):
            parse_litellm(document, checked_on=DAY)

    def test_the_table_records_the_day_it_was_read(self, trimmed: RateTable) -> None:
        assert trimmed.checked_on == DAY


_NUMBERS = st.one_of(
    st.floats(allow_nan=True, allow_infinity=True),
    st.integers(min_value=-(10**400), max_value=10**400),
    st.sampled_from([0, 1e-6, 4e-6, 2.5e-7, True, False, None, "1e-6", [], {}]),
)
_COLUMNS = st.sampled_from(
    [
        "input_cost_per_token",
        "output_cost_per_token",
        "cache_read_input_token_cost",
        "cache_creation_input_token_cost",
        "cache_creation_input_token_cost_above_1hr",
        "input_cost_per_token_priority",
        "output_cost_per_token_priority",
        "input_cost_per_token_ultrafast",
        "output_cost_per_token_ultrafast",
        "provider_specific_entry",
    ]
)
_ROWS = st.dictionaries(_COLUMNS, st.one_of(_NUMBERS, st.fixed_dictionaries({"fast": _NUMBERS})))
_NAMES = st.sampled_from(["a", "B ", " c/a", "d/e/a", "c/b", "", "x/", "/y", "A"])
_DOCUMENTS = st.dictionaries(_NAMES, st.one_of(_ROWS, _NUMBERS))
_JSON = st.recursive(
    st.one_of(st.none(), st.booleans(), st.integers(), st.floats(), st.text(max_size=8)),
    lambda inner: st.one_of(
        st.lists(inner, max_size=4), st.dictionaries(st.text(max_size=8), inner, max_size=4)
    ),
    max_leaves=12,
)


class TestParserProperties:
    @given(st.one_of(_DOCUMENTS, _JSON))
    def test_it_raises_only_the_documented_error_on_any_json(self, document: object) -> None:
        try:
            table = _parse(document)
        except RateDocumentError:
            return
        assert len(table) > 0

    @given(st.binary(max_size=64))
    def test_it_raises_only_the_documented_error_on_any_bytes(self, document: bytes) -> None:
        with contextlib.suppress(RateDocumentError):
            parse_litellm(document, checked_on=DAY)

    @given(_DOCUMENTS)
    def test_every_rate_it_returns_is_finite_and_non_negative(self, document: object) -> None:
        try:
            table = _parse(document)
        except RateDocumentError:
            return
        for key, entry in table.entries.items():
            assert key == entry.name
            for rates in (entry.standard, entry.fast, entry.ultrafast):
                if rates is not None:
                    assert all(
                        value >= 0 and value < float("inf")
                        for value in (
                            rates.input_usd,
                            rates.output_usd,
                            rates.cache_read_usd,
                            rates.cache_write_usd,
                            rates.cache_write_1h_usd or 0.0,
                        )
                    )

    @given(_DOCUMENTS, st.sampled_from(["", " ", "\t"]), st.sampled_from([str.lower, str.upper]))
    def test_lookup_ignores_case_and_surrounding_space(
        self, document: object, pad: str, case: Callable[[str], str]
    ) -> None:
        try:
            table = _parse(document)
        except RateDocumentError:
            return
        for name, entry in table.entries.items():
            spelled = f"{pad}{case(name)}{pad}"
            assert table.lookup(spelled) == entry


_COUNT = st.integers(min_value=0, max_value=10**9)


@st.composite
def token_usages(draw: st.DrawFn) -> TokenUsage:
    read = draw(_COUNT)
    write = draw(_COUNT)
    plain = draw(_COUNT)
    output = draw(_COUNT)
    return TokenUsage(
        input_tokens=read + write + plain,
        cache_read_input_tokens=read,
        cache_write_input_tokens=write,
        cache_write_1h_input_tokens=draw(st.integers(min_value=0, max_value=write)),
        output_tokens=output,
        reasoning_output_tokens=draw(st.integers(min_value=0, max_value=output)),
        turns=draw(st.integers(min_value=0, max_value=10)),
    )


_RATE = st.floats(min_value=0.0, max_value=1000.0, allow_nan=False)
_RATES = st.builds(
    Rates,
    _RATE,
    _RATE,
    _RATE,
    _RATE,
    st.one_of(st.none(), _RATE),
)


def _rates_of(pricing: ModelPricing) -> Rates:
    assert pricing.reasoning_usd is None
    return Rates(
        pricing.input_usd,
        pricing.output_usd,
        pricing.cache_read_usd,
        pricing.cache_write_usd,
        pricing.cache_write_1h_usd,
    )


class TestArithmetic:
    @given(token_usages(), st.sampled_from(default_pricing().entries))
    def test_a_pinned_row_costs_the_same_as_its_rates(
        self, tokens: TokenUsage, pricing: ModelPricing
    ) -> None:
        assert price_tokens(tokens, _rates_of(pricing)).total_usd == pytest.approx(
            cost_usd(tokens, pricing), rel=1e-9, abs=1e-9
        )

    @given(token_usages(), token_usages(), _RATES)
    def test_cost_is_additive_in_tokens(self, a: TokenUsage, b: TokenUsage, rates: Rates) -> None:
        assert price_tokens(a + b, rates).total_usd == pytest.approx(
            price_tokens(a, rates).total_usd + price_tokens(b, rates).total_usd, rel=1e-9, abs=1e-9
        )

    @given(token_usages(), _RATES)
    def test_the_breakdown_sums_to_its_total(self, tokens: TokenUsage, rates: Rates) -> None:
        cost = price_tokens(tokens, rates)
        assert isinstance(cost, CostBreakdown)
        assert cost.total_usd == pytest.approx(
            cost.input_usd + cost.cache_read_usd + cost.cache_write_usd + cost.output_usd
        )

    @given(token_usages(), _RATES, st.floats(min_value=0.0, max_value=20.0, allow_nan=False))
    def test_a_fast_tier_adds_its_premium_to_the_standard_cost(
        self, tokens: TokenUsage, standard: Rates, multiple: float
    ) -> None:
        entry = RateEntry("m", standard, fast=standard.scaled(multiple))
        premium = speed_premium_usd(tokens, entry, UsageSpeed.FAST)
        assert premium + price_tokens(tokens, standard).total_usd == pytest.approx(
            price_tokens(tokens, standard.scaled(multiple)).total_usd, rel=1e-9, abs=1e-9
        )

    @given(token_usages(), _RATES)
    def test_no_premium_is_claimed_without_a_tier(self, tokens: TokenUsage, rates: Rates) -> None:
        entry = RateEntry("m", rates)
        assert speed_premium_usd(tokens, entry, UsageSpeed.STANDARD) == 0.0
        assert speed_premium_usd(tokens, entry, UsageSpeed.FAST) == 0.0
        assert entry.at(UsageSpeed.ULTRAFAST) is None

    def test_one_hour_writes_use_their_own_rate_and_default_to_the_write_rate(self) -> None:
        tokens = TokenUsage(
            input_tokens=2_000_000,
            cache_write_input_tokens=2_000_000,
            cache_write_1h_input_tokens=1_000_000,
        )
        with_1h = price_tokens(tokens, Rates(1.0, 1.0, 0.1, 1.25, 2.0))
        without = price_tokens(tokens, Rates(1.0, 1.0, 0.1, 1.25))
        assert with_1h.cache_write_usd == pytest.approx(1.25 + 2.0)
        assert without.cache_write_usd == pytest.approx(2.5)

    def test_reasoning_tokens_are_billed_once_as_output(self) -> None:
        tokens = TokenUsage(output_tokens=1_000_000, reasoning_output_tokens=400_000)
        assert price_tokens(tokens, Rates(1.0, 8.0, 0.1, 1.0)).output_usd == pytest.approx(8.0)

    def test_cache_savings_is_the_input_rate_minus_the_read_rate_on_the_read_tokens(self) -> None:
        tokens = TokenUsage(input_tokens=3_000_000, cache_read_input_tokens=2_000_000)
        assert cache_savings_usd(tokens, Rates(2.0, 8.0, 0.5, 2.0)) == pytest.approx(3.0)

    @pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
    def test_rates_reject_a_value_that_is_not_a_finite_non_negative_number(
        self, bad: float
    ) -> None:
        with pytest.raises(ValueError, match="finite and non-negative"):
            Rates(bad, 1.0, 1.0, 1.0)


class TestBundledRates:
    def test_the_pinned_table_prices_every_pinned_model_with_the_same_standard_rates(self) -> None:
        table = bundled_rates()
        for pricing in default_pricing().entries:
            entry = table.lookup(pricing.model)
            assert entry is not None, pricing.model
            assert entry.standard == _rates_of(pricing)

    def test_the_pinned_table_has_speed_tiers(self) -> None:
        table = bundled_rates()
        opus = table.lookup("claude-opus-5-5")
        assert opus is not None
        assert opus.fast == opus.standard.scaled(2)
        sol = table.lookup("gpt-6.1-sol")
        assert sol is not None
        assert sol.fast is not None
        assert sol.ultrafast is not None

    def test_it_is_dated_by_the_snapshot_the_recorded_table_was_generated_from(self) -> None:
        assert bundled_rates().checked_on == default_pricing().last_updated


def _doc(price: float) -> bytes:
    return json.dumps(
        {"m": {"input_cost_per_token": price, "output_cost_per_token": price}}
    ).encode()


GOOD_DOCUMENTS = (_doc(1e-6), _doc(2e-6))
BAD_OUTCOMES: tuple[bytes | RateFetchError, ...] = (
    b"not json",
    b"{}",
    b"[]",
    b'{"m": {"input_cost_per_token": 1}}',
    RateFetchError("timeout"),
)


class TestRead:
    def test_with_nothing_stored_it_reads_the_pinned_table(self) -> None:
        read = read_rates(FakeRateStore(), now=NOON)
        assert read.status is RateStatus.BUNDLED
        assert read.fetched_at is None
        assert read.refresh_due
        assert read.known_models == len(read.table) > 0

    def test_a_stored_document_is_used_without_a_source(self) -> None:
        store = FakeRateStore(RateSnapshot(GOOD_DOCUMENTS[0], NOON, NOON))
        read = read_rates(store, now=NOON + dt.timedelta(hours=1))
        assert read.status is RateStatus.FRESH
        assert read.known_models == 1
        assert read.table.lookup("m") is not None
        assert not read.refresh_due

    def test_an_old_document_is_cached_and_due(self) -> None:
        store = FakeRateStore(RateSnapshot(GOOD_DOCUMENTS[0], NOON, NOON))
        read = read_rates(store, now=NOON + RATES_TTL)
        assert read.status is RateStatus.CACHED
        assert read.refresh_due

    def test_a_stored_document_that_does_not_parse_is_ignored_in_favor_of_the_pinned_table(
        self,
    ) -> None:
        store = FakeRateSource  # noqa: F841 - documents that a source is never needed to read
        snapshot = RateSnapshot(b"not json", NOON, NOON)
        read = read_rates(FakeRateStore(snapshot), now=NOON)
        assert read.status is RateStatus.BUNDLED


class TestRefresh:
    def test_first_use_fetches_and_stores_the_document(self) -> None:
        store, source = FakeRateStore(), FakeRateSource(GOOD_DOCUMENTS[0])
        read = refresh_rates(store, source, now=NOON)
        assert read.status is RateStatus.FRESH
        assert read.fetched_at == NOON
        assert store.read() == RateSnapshot(GOOD_DOCUMENTS[0], NOON, NOON)
        assert source.requests == [LITELLM_ENDPOINTS[0]]

    def test_the_second_endpoint_is_tried_when_the_first_fails(self) -> None:
        source = FakeRateSource(RateFetchError("down"), GOOD_DOCUMENTS[0])
        read = refresh_rates(FakeRateStore(), source, now=NOON)
        assert read.status is RateStatus.FRESH
        assert source.requests == list(LITELLM_ENDPOINTS)

    def test_a_document_that_parses_to_nothing_is_a_failed_fetch(self) -> None:
        store = FakeRateStore(RateSnapshot(GOOD_DOCUMENTS[0], NOON, NOON))
        later = NOON + RATES_TTL
        read = refresh_rates(store, FakeRateSource(b"{}", b"{}"), now=later)
        assert read.status is RateStatus.CACHED
        assert store.read() == RateSnapshot(GOOD_DOCUMENTS[0], NOON, later)

    def test_nothing_is_fetched_while_the_document_is_current(self) -> None:
        store = FakeRateStore(RateSnapshot(GOOD_DOCUMENTS[0], NOON, NOON))
        source = FakeRateSource(GOOD_DOCUMENTS[1])
        read = refresh_rates(store, source, now=NOON + RATES_TTL - dt.timedelta(seconds=1))
        assert source.requests == []
        assert read.status is RateStatus.FRESH

    def test_force_skips_the_ttl_but_not_the_floor(self) -> None:
        store = FakeRateStore(RateSnapshot(GOOD_DOCUMENTS[0], NOON, NOON))
        source = FakeRateSource(GOOD_DOCUMENTS[1])
        refresh_rates(store, source, now=NOON + REFRESH_FLOOR - dt.timedelta(seconds=1), force=True)
        assert source.requests == []
        refresh_rates(store, source, now=NOON + REFRESH_FLOOR, force=True)
        assert source.requests == [LITELLM_ENDPOINTS[0]]
        assert store.read() == RateSnapshot(
            GOOD_DOCUMENTS[1], NOON + REFRESH_FLOOR, NOON + REFRESH_FLOOR
        )

    def test_the_floor_holds_after_a_failure(self) -> None:
        store, source = FakeRateStore(), FakeRateSource()
        refresh_rates(store, source, now=NOON)
        assert len(source.requests) == len(LITELLM_ENDPOINTS)
        refresh_rates(store, source, now=NOON + REFRESH_FLOOR - dt.timedelta(seconds=1))
        assert len(source.requests) == len(LITELLM_ENDPOINTS)
        refresh_rates(store, source, now=NOON + REFRESH_FLOOR)
        assert len(source.requests) == 2 * len(LITELLM_ENDPOINTS)

    def test_a_failed_first_fetch_leaves_the_pinned_table_in_use(self) -> None:
        read = refresh_rates(FakeRateStore(), FakeRateSource(), now=NOON)
        assert read.status is RateStatus.BUNDLED
        assert not read.refresh_due


_STEPS = st.lists(
    st.tuples(
        st.integers(min_value=0, max_value=3 * 86_400),
        st.booleans(),
        st.sampled_from([*GOOD_DOCUMENTS, *BAD_OUTCOMES]),
    ),
    max_size=25,
)


class TestRefreshProperties:
    @given(_STEPS)
    def test_a_good_document_is_never_replaced_by_a_bad_one_and_the_policy_holds(
        self, steps: list[tuple[int, bool, bytes | RateFetchError]]
    ) -> None:
        store = FakeRateStore()
        now = NOON
        for advance, force, outcome in steps:
            now += dt.timedelta(seconds=advance)
            before = store.read()
            source = FakeRateSource(outcome, outcome)
            read = refresh_rates(store, source, now=now, force=force)
            after = store.read()

            if before is not None and before.document is not None:
                assert after is not None
                assert after.document in GOOD_DOCUMENTS
            attempted = bool(source.requests)
            if attempted:
                assert (
                    before is None
                    or before.last_attempt_at is None
                    or (now - before.last_attempt_at >= REFRESH_FLOOR)
                )
                assert (
                    force
                    or before is None
                    or before.fetched_at is None
                    or (now - before.fetched_at >= RATES_TTL)
                )
                assert after is not None
                assert after.last_attempt_at == now
            else:
                assert after == before

            if after is None or after.document is None:
                assert read.status is RateStatus.BUNDLED
            else:
                assert after.fetched_at is not None
                current = (
                    now - after.fetched_at < RATES_TTL and after.last_attempt_at == after.fetched_at
                )
                assert read.status is (RateStatus.FRESH if current else RateStatus.CACHED)


class TestFakeRateStore(RateStoreContract):
    def make_store(self) -> FakeRateStore:
        return FakeRateStore()
