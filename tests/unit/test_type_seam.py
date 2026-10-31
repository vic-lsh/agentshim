# pyright: reportUnnecessaryTypeIgnoreComment=true
"""A fetched rate cannot reach a recorded cost, and a recorded price cannot be an estimate rate.

This is checked by the type checker, not at run time: every call below is a type
error, and each carries the ignore that says so. With
``reportUnnecessaryTypeIgnoreComment`` on for this file, pyright reports an
ignore that no longer hides an error, so widening ``cost_usd``, ``price_for`` or
``price_tokens`` to accept the other kind of rate fails ``scripts/type_check.sh``.
"""

from __future__ import annotations

from agentshim import (
    ModelPricing,
    RateEntry,
    Rates,
    RateTable,
    TokenUsage,
    cost_usd,
    price_for,
    price_tokens,
)


def _a_fetched_rate_cannot_be_a_recorded_cost(
    tokens: TokenUsage, rates: Rates, entry: RateEntry, table: RateTable
) -> None:
    cost_usd(tokens, rates)  # pyright: ignore[reportArgumentType]
    cost_usd(tokens, entry)  # pyright: ignore[reportArgumentType]
    price_for("claude", "claude-opus-5-5", table)  # pyright: ignore[reportArgumentType]


def _a_recorded_price_cannot_be_an_estimate_rate(tokens: TokenUsage, pricing: ModelPricing) -> None:
    price_tokens(tokens, pricing)  # pyright: ignore[reportArgumentType]


def test_the_seam_is_held_by_the_type_checker() -> None:
    assert callable(_a_fetched_rate_cannot_be_a_recorded_cost)
    assert callable(_a_recorded_price_cannot_be_an_estimate_rate)
