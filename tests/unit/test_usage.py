"""Token accounting value objects."""

from __future__ import annotations

from agentshim import ProviderUsage, TokenUsage


def test_addition_sums_every_field() -> None:
    first = TokenUsage(
        input_tokens=1,
        output_tokens=2,
        cached_input_tokens=1,
        cache_write_input_tokens=1,
        reasoning_output_tokens=3,
        turns=1,
    )
    second = TokenUsage(
        input_tokens=3,
        output_tokens=4,
        cached_input_tokens=2,
        cache_write_input_tokens=1,
        reasoning_output_tokens=1,
        turns=1,
    )
    assert first + second == TokenUsage(
        input_tokens=4,
        output_tokens=6,
        cached_input_tokens=3,
        cache_write_input_tokens=2,
        reasoning_output_tokens=4,
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
    }


def test_provider_usage_to_dict_adds_cost_and_provider() -> None:
    usage = ProviderUsage(tokens=TokenUsage(turns=1), total_cost_usd=0.5, provider="claude")
    payload = usage.to_dict()
    assert payload["total_cost_usd"] == 0.5
    assert payload["provider"] == "claude"
    assert payload["turns"] == 1


def test_raw_defaults_to_none() -> None:
    assert ProviderUsage().raw is None
