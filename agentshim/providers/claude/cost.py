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

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, cast

from agentshim.core.pricing import cost_usd
from agentshim.core.rates import candidate_names
from agentshim.core.usage import CostSource, ModelUsage, TokenUsage, UsageSpeed, normalized_usage
from agentshim.pricing import price_for

from .model_data import SYNTHETIC_MODEL

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from agentshim.core.pricing import PricingTable

PROVIDER_NAME = "claude"
#: What a ``modelUsage`` key with no name is called: its tokens are real, but no price can match.
UNNAMED_MODEL = "unknown"


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
    #: Thinking tokens, the part of the output the CLI reports as reasoning; 0 when it does not.
    reasoning_tokens: int = 0
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
            reasoning_tokens=max(self.reasoning_tokens - baseline.reasoning_tokens, 0),
            canonical=self.canonical,
        )

    def plus(self, other: ModelCounts) -> ModelCounts:
        """These counts and *other*'s added together, naming the model as this one does."""
        return ModelCounts(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            canonical=self.canonical or other.canonical,
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
    """A turn's cost and where it came from; ``usd`` is ``None`` for an unknown model.

    ``by_model`` has a row for every model that had tokens in the turn, priced
    or not, in the order the CLI listed them. When none is unpriced the rows'
    ``recorded_usd``, added in order, are ``usd``. No row is primary yet and none has a speed:
    the stream knows which model the turn ran, the cost does not.
    """

    usd: float | None
    source: CostSource
    by_model: tuple[ModelUsage, ...] = ()


def parse_model_usage(raw: object) -> dict[str, ModelCounts] | None:
    """Read a ``result`` frame's ``modelUsage``; ``None`` when it is absent or malformed.

    A key with no name (empty or blank) is a CLI fault, not a reason to fail the turn: its
    tokens are kept under :data:`UNNAMED_MODEL`, which has no price, so the turn is
    ``UNKNOWN_MODEL`` with no cost as it is for any other model agentshim cannot price.
    """
    if not isinstance(raw, dict):
        return None
    parsed: dict[str, ModelCounts] = {}
    for model, counts in cast("dict[object, object]", raw).items():
        if not isinstance(model, str) or not isinstance(counts, dict):
            return None
        fields = cast("Mapping[str, Any]", counts)
        canonical = fields.get("canonicalModel")
        read = ModelCounts(
            input_tokens=_count(fields, "inputTokens"),
            output_tokens=_count(fields, "outputTokens"),
            cache_read_tokens=_count(fields, "cacheReadInputTokens"),
            cache_write_tokens=_count(fields, "cacheCreationInputTokens"),
            reasoning_tokens=_count(fields, "thinkingTokens"),
            canonical=canonical if isinstance(canonical, str) else None,
        )
        name = model if model.strip() else UNNAMED_MODEL
        parsed[name] = parsed[name].plus(read) if name in parsed else read
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
    ``UNKNOWN_MODEL`` with no cost: a partial sum would read as complete. The
    CLI's own placeholder for a message it wrote itself bills nothing and has
    no row.
    """
    total: float | None = 0.0
    rows: list[ModelUsage] = []
    for model, counts in model_usage.items():
        delta = counts.since(baseline.get(model))
        if delta.empty or model == SYNTHETIC_MODEL:
            continue
        pricing = price_for(PROVIDER_NAME, model, table)
        if pricing is None and delta.canonical is not None:
            pricing = price_for(PROVIDER_NAME, delta.canonical, table)
        # Cost is computed from the counts alone: splitting thinking out of the output would
        # price the same tokens in a different float order, and recorded cost must not move.
        cost = None if pricing is None else cost_usd(_tokens(delta, one_hour_share), pricing)
        total = None if total is None or cost is None else total + cost
        rows.append(
            ModelUsage(
                model=model,
                tokens=_tokens(delta, one_hour_share, with_reasoning=True),
                recorded_usd=cost,
            )
        )
    source = CostSource.PRICE_TABLE if total is not None else CostSource.UNKNOWN_MODEL
    return TurnCost(total, source, tuple(rows))


def mark_primary(
    rows: Sequence[ModelUsage],
    launched: Iterable[str | None],
    *,
    speed: UsageSpeed | None,
    turns: int,
) -> tuple[ModelUsage, ...]:
    """Name one row the model the turn ran on, and give it the turn's speed and turn count.

    The primary is the row for the first of *launched* (the models the turn
    was started with, most trustworthy first) that a row answers to, where a
    dated snapshot, a bracketed variant and a ``vendor/`` prefix are the same
    model. If none matches, it is the row with the most output tokens, the
    first of them on a tie. Claude reports one speed per turn, not per model,
    so only the primary row gets it. A turn with no rows has no primary.
    """
    if not rows:
        return ()
    chosen = next(
        (
            index
            for name in launched
            if name
            for index, row in enumerate(rows)
            if set(candidate_names(name)) & set(candidate_names(row.model))
        ),
        max(range(len(rows)), key=lambda index: (rows[index].tokens.output_tokens, -index)),
    )
    return tuple(
        replace(row, tokens=replace(row.tokens, turns=turns), speed=speed, primary=True)
        if index == chosen
        else row
        for index, row in enumerate(rows)
    )


def report_speed(usage: Mapping[str, Any] | None) -> UsageSpeed | None:
    """The speed a ``usage`` mapping says the request ran at.

    ``None`` when it says none, or names a speed agentshim has no
    :class:`UsageSpeed` for: the caller then reads it as standard, and the
    figure is a list price of the standard tier whatever was billed.
    """
    speed = None if usage is None else usage.get("speed")
    return next((tier for tier in UsageSpeed if tier.value == speed), None)


def _tokens(
    counts: ModelCounts, one_hour_share: float, *, with_reasoning: bool = False
) -> TokenUsage:
    share = min(max(one_hour_share, 0.0), 1.0)
    return normalized_usage(
        input_tokens=counts.input_tokens + counts.cache_read_tokens + counts.cache_write_tokens,
        output_tokens=counts.output_tokens,
        cache_read_input_tokens=counts.cache_read_tokens,
        cache_write_input_tokens=counts.cache_write_tokens,
        cache_write_1h_input_tokens=round(counts.cache_write_tokens * share),
        reasoning_output_tokens=counts.reasoning_tokens if with_reasoning else 0,
    )


def _count(counts: Mapping[str, Any], key: str) -> int:
    value = counts.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return max(int(value), 0)
