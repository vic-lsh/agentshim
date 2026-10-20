"""Price a Claude turn from the CLI's per-model token counts.

The CLI's own ``total_cost_usd`` is priced by the CLI's built-in table. A CLI
older than the model it runs does not know the model and bills it at a
fallback rate, so that figure can be several times too high. The ``modelUsage``
mapping of a ``result`` frame carries the token counts per model, which
agentshim prices with its own table instead.

``modelUsage`` is cumulative over the CLI process, like ``total_cost_usd``, so
a turn's tokens are the difference from the previous result's counts. It does
not split cache writes by TTL; the turn's ``usage.cache_creation`` does, and
its share of one-hour writes is applied to every model's writes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from agentshim.core.pricing import cost_usd
from agentshim.core.usage import CostSource, TokenUsage, normalized_usage
from agentshim.pricing import price_for

if TYPE_CHECKING:
    from collections.abc import Mapping

    from agentshim.core.pricing import PricingTable

PROVIDER_NAME = "claude"


@dataclass(frozen=True)
class ModelCounts:
    """One model's cumulative token counts as ``modelUsage`` reports them.

    ``input_tokens`` is uncached input, disjoint from the cache counts, as in
    the CLI's own report.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    #: The id the CLI says the model canonically is, when it says so.
    canonical: str | None = None

    def since(self, baseline: ModelCounts | None) -> ModelCounts:
        """The counts accrued after *baseline*; a count that went down reads as 0."""
        if baseline is None:
            return self
        return ModelCounts(
            input_tokens=max(self.input_tokens - baseline.input_tokens, 0),
            output_tokens=max(self.output_tokens - baseline.output_tokens, 0),
            cache_read_tokens=max(self.cache_read_tokens - baseline.cache_read_tokens, 0),
            cache_write_tokens=max(self.cache_write_tokens - baseline.cache_write_tokens, 0),
            canonical=self.canonical,
        )

    @property
    def empty(self) -> bool:
        """Whether no token was counted."""
        return not (
            self.input_tokens
            or self.output_tokens
            or self.cache_read_tokens
            or self.cache_write_tokens
        )


@dataclass(frozen=True)
class TurnCost:
    """A turn's cost and where it came from; ``usd`` is ``None`` for an unknown model."""

    usd: float | None
    source: CostSource


def parse_model_usage(raw: object) -> dict[str, ModelCounts] | None:
    """Read a ``result`` frame's ``modelUsage``; ``None`` when it is absent or malformed."""
    if not isinstance(raw, dict):
        return None
    parsed: dict[str, ModelCounts] = {}
    for model, counts in cast("dict[object, object]", raw).items():
        if not isinstance(model, str) or not isinstance(counts, dict):
            return None
        fields = cast("Mapping[str, Any]", counts)
        canonical = fields.get("canonicalModel")
        parsed[model] = ModelCounts(
            input_tokens=_count(fields, "inputTokens"),
            output_tokens=_count(fields, "outputTokens"),
            cache_read_tokens=_count(fields, "cacheReadInputTokens"),
            cache_write_tokens=_count(fields, "cacheCreationInputTokens"),
            canonical=canonical if isinstance(canonical, str) else None,
        )
    return parsed


def priced_turn(
    model_usage: Mapping[str, ModelCounts],
    baseline: Mapping[str, ModelCounts],
    *,
    one_hour_share: float,
    table: PricingTable | None = None,
) -> TurnCost:
    """Price one turn's tokens, each model at its own table price.

    *one_hour_share* is the fraction of cache writes billed at the one-hour
    rate. A model with tokens in the turn and no price makes the whole turn
    ``UNKNOWN_MODEL`` with no cost: a partial sum would read as complete.
    """
    total = 0.0
    for model, counts in model_usage.items():
        delta = counts.since(baseline.get(model))
        if delta.empty:
            continue
        pricing = price_for(PROVIDER_NAME, model, table)
        if pricing is None and delta.canonical is not None:
            pricing = price_for(PROVIDER_NAME, delta.canonical, table)
        if pricing is None:
            return TurnCost(None, CostSource.UNKNOWN_MODEL)
        total += cost_usd(_tokens(delta, one_hour_share), pricing)
    return TurnCost(total, CostSource.PRICE_TABLE)


def _tokens(counts: ModelCounts, one_hour_share: float) -> TokenUsage:
    share = min(max(one_hour_share, 0.0), 1.0)
    return normalized_usage(
        input_tokens=counts.input_tokens + counts.cache_read_tokens + counts.cache_write_tokens,
        output_tokens=counts.output_tokens,
        cache_read_input_tokens=counts.cache_read_tokens,
        cache_write_input_tokens=counts.cache_write_tokens,
        cache_write_1h_input_tokens=round(counts.cache_write_tokens * share),
    )


def _count(counts: Mapping[str, Any], key: str) -> int:
    value = counts.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return max(int(value), 0)
