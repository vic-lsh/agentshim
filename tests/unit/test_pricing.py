"""The static pricing table and cost helpers."""

from __future__ import annotations

import dataclasses
import datetime as dt
from typing import Any

import pytest
from agentshim import (
    ModelPricing,
    ModelSpec,
    PricingTable,
    TokenUsage,
    cost_usd,
    default_models,
    default_pricing,
    get_provider,
    price_for,
    provider_names,
)

TABLE = default_pricing()
CHECKED = dt.date(2026, 9, 27)


def _entry(**overrides: object) -> ModelPricing:
    fields: dict[str, Any] = {
        "vendor": "openai",
        "model": "test-model",
        "input_usd": 1.0,
        "cache_read_usd": 0.1,
        "cache_write_usd": 1.0,
        "output_usd": 8.0,
        "source": "https://example.com/pricing",
        "checked": CHECKED,
    }
    fields.update(overrides)
    return ModelPricing(**fields)


class TestShippedTable:
    def test_it_is_versioned_and_dated(self) -> None:
        assert TABLE.version.strip()
        assert isinstance(TABLE.last_updated, dt.date)

    @pytest.mark.parametrize(
        "entry", TABLE.entries, ids=lambda entry: f"{entry.vendor}/{entry.model}"
    )
    def test_every_entry_names_its_source_and_check_date(self, entry: ModelPricing) -> None:
        assert entry.source.startswith("https://")
        assert isinstance(entry.checked, dt.date)
        assert entry.checked <= TABLE.last_updated
        assert not entry.estimated or entry.note.strip()

    @pytest.mark.parametrize(
        ("provider", "model"),
        [
            ("codex", "gpt-6-luna"),
            ("codex", "gpt-6-astra"),
            ("codex", "gpt-5.5"),
            ("codex", "gpt-5.4"),
            ("codex", "gpt-5"),
            ("claude", "claude-opus-5-5"),
            ("claude", "claude-sonnet-5"),
            ("claude", "claude-fable-5-1"),
            ("claude", "claude-haiku-4-5"),
        ],
    )
    def test_the_models_we_run_are_priced(self, provider: str, model: str) -> None:
        assert price_for(provider, model) is not None

    def test_cache_reads_cost_less_than_base_input(self) -> None:
        for entry in TABLE.entries:
            assert entry.cache_read_usd < entry.input_usd, entry.model

    def test_it_round_trips_through_a_mapping(self) -> None:
        assert PricingTable.from_dict(TABLE.to_dict()) == TABLE


class TestRequiredProvenance:
    @pytest.mark.parametrize("source", ["", "example.com/pricing", "http://example.com"])
    def test_an_entry_needs_an_https_source(self, source: str) -> None:
        with pytest.raises(ValueError, match="source"):
            _entry(source=source)

    def test_an_entry_needs_a_check_date(self) -> None:
        with pytest.raises(TypeError, match="checked"):
            _entry(checked=None)

    def test_a_mapping_without_source_or_date_fails(self) -> None:
        row = _entry().to_dict()
        for key in ("source", "checked"):
            with pytest.raises(ValueError, match=key):
                ModelPricing.from_dict({k: v for k, v in row.items() if k != key})

    def test_a_table_needs_last_updated(self) -> None:
        with pytest.raises(ValueError, match="last_updated"):
            PricingTable.from_dict({"version": "x", "entries": []})

    def test_no_entry_may_postdate_the_table(self) -> None:
        with pytest.raises(ValueError, match="last_updated"):
            PricingTable(
                version="x",
                last_updated=CHECKED - dt.timedelta(days=1),
                entries=(_entry(),),
            )

    def test_an_estimate_must_state_its_basis(self) -> None:
        with pytest.raises(ValueError, match="basis"):
            _entry(estimated=True)

    def test_duplicate_keys_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicate"):
            PricingTable(version="x", last_updated=CHECKED, entries=(_entry(), _entry()))


class TestLookup:
    def test_an_unknown_model_is_none_not_zero(self) -> None:
        assert price_for("codex", "gpt-99-imaginary") is None

    def test_an_unpriced_provider_is_none(self) -> None:
        assert price_for("copilot", "gpt-5") is None
        assert price_for("codex", None) is None

    @pytest.mark.parametrize(
        ("provider", "model", "expected"),
        [
            ("claude", "claude-haiku-4-5-20251001", "claude-haiku-4-5"),
            ("claude", "claude-opus-5-5[1m]", "claude-opus-5-5"),
            ("opencode", "anthropic/claude-sonnet-5", "claude-sonnet-5"),
            ("opencode", "openai/gpt-5.4", "gpt-5.4"),
            ("codex", "GPT-6-Luna", "gpt-6-luna"),
        ],
    )
    def test_names_resolve_to_table_keys(self, provider: str, model: str, expected: str) -> None:
        entry = price_for(provider, model)
        assert entry is not None
        assert entry.model == expected

    def test_a_custom_table_overrides_and_extends(self) -> None:
        pinned = _entry(model="gpt-6-luna", input_usd=0.2)
        extra = _entry(model="in-house")
        table = TABLE.with_entries([pinned, extra], version="experiment-1")
        assert table.version == "experiment-1"
        assert price_for("codex", "gpt-6-luna", table) == pinned
        assert price_for("codex", "in-house", table) == extra
        assert price_for("codex", "gpt-6-luna") != pinned


class TestCost:
    def test_claude_cost_prices_each_class_once(self) -> None:
        pricing = price_for("claude", "claude-haiku-4-5")
        assert pricing is not None
        usage = TokenUsage(
            input_tokens=1_000_000 * 4,
            cache_read_input_tokens=1_000_000,
            cache_write_input_tokens=2_000_000,
            cache_write_1h_input_tokens=1_000_000,
            output_tokens=1_000_000,
        )
        assert cost_usd(usage, pricing) == pytest.approx(1.00 + 0.10 + 1.25 + 2.00 + 5.00)

    def test_codex_reasoning_is_billed_as_output(self) -> None:
        pricing = price_for("codex", "gpt-6-luna")
        assert pricing is not None
        with_reasoning = TokenUsage(output_tokens=1_000_000, reasoning_output_tokens=600_000)
        without = TokenUsage(output_tokens=1_000_000)
        assert (
            cost_usd(with_reasoning, pricing) == cost_usd(without, pricing) == pytest.approx(0.50)
        )

    def test_separately_priced_reasoning_uses_its_rate(self) -> None:
        pricing = _entry(reasoning_usd=20.0)
        usage = TokenUsage(output_tokens=1_000_000, reasoning_output_tokens=500_000)
        assert cost_usd(usage, pricing) == pytest.approx(0.5 * 8 + 0.5 * 20)

    def test_relative_weights_are_multiples_of_base_input(self) -> None:
        pricing = price_for("claude", "claude-sonnet-5")
        assert pricing is not None
        weights = pricing.relative_weights()
        assert weights.uncached_input == 1.0
        assert weights.cache_read_input == pytest.approx(0.1)
        assert weights.cache_write_input == pytest.approx(1.25)
        assert weights.cache_write_1h_input == pytest.approx(2.0)
        assert weights.output == pytest.approx(5.0)

    def test_usd_and_relative_weights_agree(self) -> None:
        pricing = price_for("codex", "gpt-6-luna")
        assert pricing is not None
        usage = TokenUsage(input_tokens=1000, cache_read_input_tokens=900, output_tokens=50)
        relative = usage.weighted_total(pricing.relative_weights())
        assert cost_usd(usage, pricing) == pytest.approx(relative * pricing.input_usd / 1_000_000)


def test_entries_are_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        TABLE.entries[0].input_usd = 0  # type: ignore[misc]


#: Models the catalogue lists that the table deliberately leaves unpriced, with why.
#: A model absent here must have a price, so a new catalogue model cannot ship without one.
UNPRICED_BY_DESIGN = {
    "gemini-3-pro-preview": (
        "Google's pricing page no longer lists it, so no official number can be checked; "
        "Gemini models are not priced."
    ),
}


def _unpriced_by_design(model: str) -> bool:
    return model.rpartition("/")[2] in UNPRICED_BY_DESIGN


@pytest.mark.parametrize(
    "spec", default_models().entries, ids=lambda spec: f"{spec.vendor}/{spec.model}"
)
def test_every_catalogue_model_is_priced_or_excluded_by_design(spec: ModelSpec) -> None:
    priced = price_for(spec.vendor, spec.model) is not None
    assert priced != _unpriced_by_design(spec.model), (
        f"{spec.vendor}/{spec.model} is in the model catalogue but "
        + ("is listed as unpriced and has a price" if priced else "has no price")
    )


@pytest.mark.parametrize("provider", provider_names())
def test_every_model_a_provider_offers_is_priced_or_excluded_by_design(provider: str) -> None:
    for model in get_provider(provider).profile.models:
        priced = price_for(provider, model) is not None
        assert priced != _unpriced_by_design(model), (provider, model)


def test_every_exclusion_names_a_catalogue_model() -> None:
    catalogued = {spec.model for spec in default_models().entries}
    assert set(UNPRICED_BY_DESIGN) <= catalogued


class TestGpt61Sol:
    """Regression: the model was in the catalogue with no price, so its turns recorded no cost."""

    def test_it_is_priced_at_the_published_standard_rates(self) -> None:
        pricing = price_for("codex", "gpt-6.1-sol")
        assert pricing is not None
        assert (
            pricing.input_usd,
            pricing.cache_read_usd,
            pricing.cache_write_usd,
            pricing.output_usd,
        ) == (2.00, 0.10, 2.50, 10.00)
        assert not pricing.estimated
        assert pricing.checked >= dt.date(2026, 10, 30)

    def test_its_cache_read_rate_is_not_gpt_6_sols(self) -> None:
        newer, older = price_for("codex", "gpt-6.1-sol"), price_for("codex", "gpt-6-sol")
        assert newer is not None
        assert older is not None
        assert newer.cache_read_usd != older.cache_read_usd

    def test_a_turn_costs_each_token_class_once(self) -> None:
        pricing = price_for("codex", "gpt-6.1-sol")
        assert pricing is not None
        usage = TokenUsage(
            input_tokens=3_000_000,
            cache_read_input_tokens=1_000_000,
            cache_write_input_tokens=1_000_000,
            output_tokens=1_000_000,
        )
        assert cost_usd(usage, pricing) == pytest.approx(2.00 + 0.10 + 2.50 + 10.00)
