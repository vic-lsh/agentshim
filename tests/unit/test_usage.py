"""Token accounting value objects."""

from __future__ import annotations

import pytest
from agentshim import ProviderUsage, TokenUsage, TokenWeights, normalized_usage
from hypothesis import given
from hypothesis import strategies as st


def _usage(**counts: int) -> TokenUsage:
    return TokenUsage(**counts)


def test_addition_sums_every_field() -> None:
    first = TokenUsage(
        input_tokens=10,
        output_tokens=5,
        cache_read_input_tokens=4,
        cache_write_input_tokens=3,
        cache_write_1h_input_tokens=1,
        reasoning_output_tokens=2,
        turns=1,
    )
    second = TokenUsage(
        input_tokens=20,
        output_tokens=6,
        cache_read_input_tokens=8,
        cache_write_input_tokens=2,
        cache_write_1h_input_tokens=2,
        reasoning_output_tokens=1,
        turns=1,
    )
    assert first + second == TokenUsage(
        input_tokens=30,
        output_tokens=11,
        cache_read_input_tokens=12,
        cache_write_input_tokens=5,
        cache_write_1h_input_tokens=3,
        reasoning_output_tokens=3,
        turns=2,
    )


def test_token_usage_to_dict_keys() -> None:
    assert TokenUsage(input_tokens=1, turns=4).to_dict() == {
        "input_tokens": 1,
        "output_tokens": 0,
        "cached_input_tokens": 0,
        "cache_write_input_tokens": 0,
        "reasoning_output_tokens": 0,
        "turns": 4,
        "cache_read_input_tokens": 0,
        "cache_write_1h_input_tokens": 0,
        "uncached_input_tokens": 1,
    }


def test_to_dict_only_adds_keys_to_the_0_6_shape() -> None:
    legacy = {
        "input_tokens",
        "output_tokens",
        "cached_input_tokens",
        "cache_write_input_tokens",
        "reasoning_output_tokens",
        "turns",
    }
    assert legacy <= set(TokenUsage().to_dict())


class TestCachedAlias:
    def test_cached_input_tokens_mirrors_cache_reads(self) -> None:
        usage = TokenUsage(input_tokens=10, cache_read_input_tokens=4)
        assert usage.cached_input_tokens == 4

    def test_the_deprecated_keyword_still_constructs_cache_reads(self) -> None:
        usage = TokenUsage(input_tokens=10, cached_input_tokens=4)
        assert usage.cache_read_input_tokens == 4
        assert usage == TokenUsage(input_tokens=10, cache_read_input_tokens=4)

    def test_disagreeing_alias_and_field_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="alias"):
            TokenUsage(input_tokens=10, cached_input_tokens=4, cache_read_input_tokens=5)


class TestInvariants:
    def test_uncached_input_is_what_the_cache_did_not_touch(self) -> None:
        usage = TokenUsage(
            input_tokens=100, cache_read_input_tokens=60, cache_write_input_tokens=30
        )
        assert usage.uncached_input_tokens == 10

    @pytest.mark.parametrize(
        ("counts", "match"),
        [
            (
                {"input_tokens": 10, "cache_read_input_tokens": 6, "cache_write_input_tokens": 5},
                "exceed",
            ),
            (
                {
                    "input_tokens": 10,
                    "cache_write_input_tokens": 2,
                    "cache_write_1h_input_tokens": 3,
                },
                "1h",
            ),
            ({"output_tokens": 1, "reasoning_output_tokens": 2}, "reasoning"),
            ({"input_tokens": -1}, "negative"),
        ],
    )
    def test_an_impossible_breakdown_is_rejected(self, counts: dict[str, int], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            _usage(**counts)

    def test_a_non_int_count_is_rejected(self) -> None:
        with pytest.raises(TypeError):
            TokenUsage(input_tokens=1.5)  # type: ignore[arg-type]

    def test_normalized_usage_clamps_instead_of_raising(self) -> None:
        usage = normalized_usage(
            input_tokens=10,
            output_tokens=3,
            cache_read_input_tokens=8,
            cache_write_input_tokens=8,
            cache_write_1h_input_tokens=9,
            reasoning_output_tokens=5,
        )
        assert (usage.cache_read_input_tokens, usage.cache_write_input_tokens) == (8, 2)
        assert usage.cache_write_1h_input_tokens == 2
        assert usage.reasoning_output_tokens == 3

    @given(st.lists(st.integers(-5, 10_000), min_size=6, max_size=6))
    def test_normalized_usage_always_satisfies_the_invariants(self, counts: list[int]) -> None:
        input_tokens, output, read, write, write_1h, reasoning = counts
        usage = normalized_usage(
            input_tokens=input_tokens,
            output_tokens=output,
            cache_read_input_tokens=read,
            cache_write_input_tokens=write,
            cache_write_1h_input_tokens=write_1h,
            reasoning_output_tokens=reasoning,
        )
        assert usage.uncached_input_tokens >= 0
        assert usage == TokenUsage.from_dict(usage.to_dict())


class TestFromDict:
    def test_round_trips(self) -> None:
        usage = TokenUsage(
            input_tokens=100,
            output_tokens=9,
            cache_read_input_tokens=50,
            cache_write_input_tokens=20,
            cache_write_1h_input_tokens=5,
            reasoning_output_tokens=4,
            turns=2,
        )
        assert TokenUsage.from_dict(usage.to_dict()) == usage

    def test_a_pre_0_7_mapping_had_cache_writes_inside_cached(self) -> None:
        legacy = {
            "input_tokens": 150,
            "output_tokens": 40,
            "cached_input_tokens": 50,
            "cache_write_input_tokens": 30,
            "turns": 4,
        }
        usage = TokenUsage.from_dict(legacy)
        assert usage.cache_read_input_tokens == 20
        assert usage.cache_write_input_tokens == 30
        assert usage.uncached_input_tokens == 100

    def test_a_codex_mapping_without_writes_reads_cached_as_reads(self) -> None:
        usage = TokenUsage.from_dict(
            {"input_tokens": 10, "cached_input_tokens": 7, "output_tokens": 1}
        )
        assert usage.cache_read_input_tokens == 7


class TestWeightedTotal:
    def test_each_token_class_is_weighted_once(self) -> None:
        usage = TokenUsage(
            input_tokens=100,
            output_tokens=20,
            cache_read_input_tokens=50,
            cache_write_input_tokens=30,
            cache_write_1h_input_tokens=10,
            reasoning_output_tokens=5,
        )
        weights = TokenWeights(
            uncached_input=1.0,
            cache_read_input=0.1,
            cache_write_input=1.25,
            cache_write_1h_input=2.0,
            output=5.0,
            reasoning_output=7.0,
        )
        expected = 20 * 1.0 + 50 * 0.1 + 20 * 1.25 + 10 * 2.0 + 15 * 5.0 + 5 * 7.0
        assert usage.weighted_total(weights) == pytest.approx(expected)

    def test_optional_weights_fall_back_to_their_parent_class(self) -> None:
        usage = TokenUsage(
            input_tokens=10,
            output_tokens=4,
            cache_write_input_tokens=10,
            cache_write_1h_input_tokens=10,
            reasoning_output_tokens=4,
        )
        weights = TokenWeights(uncached_input=1, cache_read_input=0, cache_write_input=2, output=3)
        assert usage.weighted_total(weights) == 10 * 2 + 4 * 3

    def test_unit_weights_give_the_raw_total(self) -> None:
        usage = TokenUsage(input_tokens=100, output_tokens=20, cache_read_input_tokens=50)
        ones = TokenWeights(uncached_input=1, cache_read_input=1, cache_write_input=1, output=1)
        assert usage.weighted_total(ones) == 120

    @pytest.mark.parametrize("bad", [-1.0, float("inf"), float("nan")])
    def test_a_weight_must_be_finite_and_non_negative(self, bad: float) -> None:
        with pytest.raises(ValueError, match="finite"):
            TokenWeights(uncached_input=bad, cache_read_input=0, cache_write_input=0, output=0)


def test_provider_usage_to_dict_adds_cost_and_provider() -> None:
    usage = ProviderUsage(tokens=TokenUsage(turns=1), total_cost_usd=0.5, provider="claude")
    payload = usage.to_dict()
    assert payload["total_cost_usd"] == 0.5
    assert payload["provider"] == "claude"
    assert payload["turns"] == 1
    assert payload["increment_known"] is True


def test_raw_defaults_to_none() -> None:
    assert ProviderUsage().raw is None


def test_unknown_usage_marker_survives_serialization_without_exposing_raw_totals() -> None:
    usage = ProviderUsage(provider="codex", raw={"input_tokens": 200}, increment_known=False)
    payload = usage.to_dict()
    assert payload["increment_known"] is False
    assert payload["input_tokens"] == 0
    assert "raw" not in payload
