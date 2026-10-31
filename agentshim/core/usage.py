"""Token and cost accounting shared by every provider.

Every provider's counts are normalized to one breakdown of the input and
output a turn was billed for:

- ``input_tokens``: every input token, including cache reads and cache writes.
- ``cache_read_input_tokens``: input served from the provider's prompt cache.
- ``cache_write_input_tokens``: input written to the prompt cache this turn.
  ``cache_write_1h_input_tokens`` is the part written with a one-hour TTL
  (Anthropic's extended cache), which is billed above the default TTL.
- ``uncached_input_tokens`` (derived): ``input - cache_read - cache_write``,
  input billed at the base rate.
- ``output_tokens``: every generated token, reasoning included.
  ``reasoning_output_tokens`` is the part the provider reports as reasoning.

``cached_input_tokens`` is a deprecated alias of ``cache_read_input_tokens``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

_COUNT_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cached_input_tokens",
    "cache_write_input_tokens",
    "reasoning_output_tokens",
    "turns",
    "cache_read_input_tokens",
    "cache_write_1h_input_tokens",
)


@dataclass(frozen=True)
class TokenWeights:
    """Caller-supplied weight per token class, for :meth:`TokenUsage.weighted_total`.

    The classes partition a turn's tokens, so no token is weighted twice.
    ``cache_write_1h_input`` defaults to ``cache_write_input`` and
    ``reasoning_output`` to ``output`` (reasoning billed as output). The
    weights carry no unit: USD per token, per million tokens, or a multiple of
    the base input rate all work. agentshim's own prices live in
    :mod:`agentshim.pricing`.
    """

    uncached_input: float
    cache_read_input: float
    cache_write_input: float
    output: float
    cache_write_1h_input: float | None = None
    reasoning_output: float | None = None

    def __post_init__(self) -> None:
        """Reject a weight that is not a finite, non-negative number."""
        for name in (
            "uncached_input",
            "cache_read_input",
            "cache_write_input",
            "output",
            "cache_write_1h_input",
            "reasoning_output",
        ):
            value = getattr(self, name)
            if value is None and name in ("cache_write_1h_input", "reasoning_output"):
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                msg = f"{name} must be a number, got {value!r}"
                raise TypeError(msg)
            if not math.isfinite(value) or value < 0:
                msg = f"{name} must be finite and non-negative, got {value!r}"
                raise ValueError(msg)


@dataclass(frozen=True)
class TokenUsage:
    """Token counts for one turn, normalized across providers (see module docstring).

    Invariants, checked on construction:
    ``cache_read_input_tokens + cache_write_input_tokens <= input_tokens``,
    ``cache_write_1h_input_tokens <= cache_write_input_tokens`` and
    ``reasoning_output_tokens <= output_tokens``.

    ``cached_input_tokens`` is kept equal to ``cache_read_input_tokens``:
    pass either one. Before 0.7.0 it also counted cache writes on Claude,
    opencode and Copilot; :meth:`from_dict` reads such records back.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_input_tokens: int = 0
    reasoning_output_tokens: int = 0
    turns: int = 0
    cache_read_input_tokens: int = 0
    cache_write_1h_input_tokens: int = 0

    def __post_init__(self) -> None:
        """Mirror the deprecated alias and check every invariant."""
        for name in _COUNT_FIELDS:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                msg = f"{name} must be an int, got {value!r}"
                raise TypeError(msg)
            if value < 0:
                msg = f"{name} must not be negative, got {value}"
                raise ValueError(msg)
        read, alias = self.cache_read_input_tokens, self.cached_input_tokens
        if read and alias and read != alias:
            msg = (
                "cached_input_tokens is an alias of cache_read_input_tokens; "
                f"got {alias} and {read}"
            )
            raise ValueError(msg)
        read = read or alias
        object.__setattr__(self, "cache_read_input_tokens", read)
        object.__setattr__(self, "cached_input_tokens", read)
        if read + self.cache_write_input_tokens > self.input_tokens:
            msg = (
                "cache reads plus cache writes must not exceed input_tokens: "
                f"{read} + {self.cache_write_input_tokens} > {self.input_tokens}"
            )
            raise ValueError(msg)
        if self.cache_write_1h_input_tokens > self.cache_write_input_tokens:
            msg = "cache_write_1h_input_tokens must not exceed cache_write_input_tokens"
            raise ValueError(msg)
        if self.reasoning_output_tokens > self.output_tokens:
            msg = "reasoning_output_tokens must not exceed output_tokens"
            raise ValueError(msg)

    @property
    def uncached_input_tokens(self) -> int:
        """Input billed at the base rate: neither read from nor written to the cache."""
        return self.input_tokens - self.cache_read_input_tokens - self.cache_write_input_tokens

    def __add__(self, other: TokenUsage) -> TokenUsage:
        """Accumulate a running total over the turns of a session.

        Every field is summed, ``turns`` included, so folding per-turn usages
        together yields both the token totals and the turn count.
        """
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_input_tokens=self.cache_read_input_tokens + other.cache_read_input_tokens,
            cache_write_input_tokens=self.cache_write_input_tokens + other.cache_write_input_tokens,
            cache_write_1h_input_tokens=(
                self.cache_write_1h_input_tokens + other.cache_write_1h_input_tokens
            ),
            reasoning_output_tokens=self.reasoning_output_tokens + other.reasoning_output_tokens,
            turns=self.turns + other.turns,
        )

    def weighted_total(self, weights: TokenWeights) -> float:
        """Sum every token class times its weight.

        The classes partition the turn (uncached input, cache reads, default-TTL
        cache writes, one-hour cache writes, non-reasoning output, reasoning
        output), so each token is weighted exactly once.
        """
        write_1h = (
            weights.cache_write_input
            if weights.cache_write_1h_input is None
            else weights.cache_write_1h_input
        )
        reasoning = weights.output if weights.reasoning_output is None else weights.reasoning_output
        return (
            self.uncached_input_tokens * weights.uncached_input
            + self.cache_read_input_tokens * weights.cache_read_input
            + (self.cache_write_input_tokens - self.cache_write_1h_input_tokens)
            * weights.cache_write_input
            + self.cache_write_1h_input_tokens * write_1h
            + (self.output_tokens - self.reasoning_output_tokens) * weights.output
            + self.reasoning_output_tokens * reasoning
        )

    def to_dict(self) -> dict[str, int]:
        """Flatten the counts into a JSON-serializable mapping keyed by field name.

        Includes the derived ``uncached_input_tokens``. Keys are only ever
        added, so a reader of an older mapping keeps working.
        """
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "cache_write_input_tokens": self.cache_write_input_tokens,
            "reasoning_output_tokens": self.reasoning_output_tokens,
            "turns": self.turns,
            "cache_read_input_tokens": self.cache_read_input_tokens,
            "cache_write_1h_input_tokens": self.cache_write_1h_input_tokens,
            "uncached_input_tokens": self.uncached_input_tokens,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TokenUsage:
        """Read back a :meth:`to_dict` mapping, including one written before 0.7.0.

        A mapping without ``cache_read_input_tokens`` predates the split:
        its ``cached_input_tokens`` counted cache writes too (Claude, opencode,
        Copilot; Codex reported none), so the writes are subtracted back out.
        Unknown keys are ignored; missing or non-numeric counts read as 0.
        """
        write = _count(data.get("cache_write_input_tokens"))
        if "cache_read_input_tokens" in data:
            read = _count(data.get("cache_read_input_tokens"))
        else:
            read = max(_count(data.get("cached_input_tokens")) - write, 0)
        return cls(
            input_tokens=_count(data.get("input_tokens")),
            output_tokens=_count(data.get("output_tokens")),
            cache_read_input_tokens=read,
            cache_write_input_tokens=write,
            cache_write_1h_input_tokens=_count(data.get("cache_write_1h_input_tokens")),
            reasoning_output_tokens=_count(data.get("reasoning_output_tokens")),
            turns=_count(data.get("turns")),
        )


def _count(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)) and math.isfinite(value) and value > 0:
        return int(value)
    return 0


def normalized_usage(  # noqa: PLR0913 - one keyword per normalized count
    *,
    input_tokens: int,
    output_tokens: int,
    cache_read_input_tokens: int = 0,
    cache_write_input_tokens: int = 0,
    cache_write_1h_input_tokens: int = 0,
    reasoning_output_tokens: int = 0,
    turns: int = 0,
) -> TokenUsage:
    """Build a :class:`TokenUsage` from provider counts, clamping any that break an invariant.

    Parsers use this so a CLI that reports inconsistent counts (a cache read
    larger than the input, say) yields the closest valid usage instead of
    failing the turn. Negative counts read as 0.
    """
    input_total = max(input_tokens, 0)
    output_total = max(output_tokens, 0)
    read = min(max(cache_read_input_tokens, 0), input_total)
    write = min(max(cache_write_input_tokens, 0), input_total - read)
    return TokenUsage(
        input_tokens=input_total,
        output_tokens=output_total,
        cache_read_input_tokens=read,
        cache_write_input_tokens=write,
        cache_write_1h_input_tokens=min(max(cache_write_1h_input_tokens, 0), write),
        reasoning_output_tokens=min(max(reasoning_output_tokens, 0), output_total),
        turns=max(turns, 0),
    )


class UsageSpeed(str, Enum):
    """The service speed a request was billed at."""

    STANDARD = "standard"
    FAST = "fast"
    ULTRAFAST = "ultrafast"


class CostSource(str, Enum):
    """Where a turn's cost came from, or that it has none to come from.

    Every parsed turn names one: a reader that finds none is reading a usage built before the
    field existed, never a provider that left it out.
    """

    #: Computed from the turn's token counts and agentshim's price table.
    PRICE_TABLE = "price_table"
    #: The CLI's own figure, used only when it did not say which model ran.
    PROVIDER_REPORTED = "provider_reported"
    #: The turn used a model agentshim has no price for, so there is no cost.
    UNKNOWN_MODEL = "unknown_model"
    #: The provider reports no cost and agentshim computes none (a CLI that bills in another
    #: way, a resumed turn whose increment is unknown), so there is no cost and nothing to price.
    NOT_REPORTED = "not_reported"


@dataclass(frozen=True)
class ModelUsage:
    """One model's share of a turn's usage.

    A turn can run several models: Claude's Task subagents and advisors run on
    their own, and ``ProviderUsage.tokens`` leaves them out. The sum of
    :attr:`tokens` over :attr:`ProviderUsage.by_model` is the whole turn.

    Attributes:
        model: The model id as the provider reports it.
        tokens: This turn's tokens for this model. Only the primary model
            carries the turn's ``turns`` count.
        recorded_usd: The pinned-table price of ``tokens`` at the standard
            tier; ``None`` when the table has no price for the model. This is
            the figure a turn's recorded cost sums, never the CLI's own. It is
            not a bill: the pinned table has no speed tiers, so a row whose
            ``speed`` is fast or ultrafast was billed more than this. For the
            price at the tier, use :func:`agentshim.price_tokens` with the
            entry's rates at ``speed``, or add
            :func:`agentshim.speed_premium_usd`.
        speed: The service speed the turn ran at, on the primary model only;
            ``None`` when the provider did not report one (read as standard).
        primary: Whether this is the model the turn was run with. Exactly one
            row of a turn is.
    """

    model: str
    tokens: TokenUsage
    recorded_usd: float | None = None
    speed: UsageSpeed | None = None
    primary: bool = False

    def __post_init__(self) -> None:
        """Reject an empty model or a cost that is not a finite, non-negative number."""
        if not self.model.strip():
            msg = "model must be a non-empty string"
            raise ValueError(msg)
        cost = self.recorded_usd
        if cost is not None and (isinstance(cost, bool) or not math.isfinite(cost) or cost < 0):
            msg = f"recorded_usd must be finite and non-negative, got {cost!r}"
            raise ValueError(msg)

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serializable mapping of this row."""
        return {
            "model": self.model,
            "tokens": self.tokens.to_dict(),
            "recorded_usd": self.recorded_usd,
            "speed": None if self.speed is None else self.speed.value,
            "primary": self.primary,
        }


def merge_model_usage(
    first: Sequence[ModelUsage], second: Sequence[ModelUsage]
) -> tuple[ModelUsage, ...]:
    """Sum the per-model usage of two separate turns into one.

    This adds turns that each have their own rows. Rows a parser built from a
    CLI's cumulative counts already cover the whole run and must not be added.

    Rows with the same model and speed add up, and a sum is unpriced if either
    part was. The primary row of *first* (else of *second*) stays the only
    primary, so a merged run still has exactly one.
    """
    rows = (*first, *second)
    primary = next((row for row in rows if row.primary), None)
    merged: dict[tuple[str, UsageSpeed | None], ModelUsage] = {}
    for row in rows:
        key = (row.model, row.speed)
        held = merged.get(key)
        if held is None:
            merged[key] = row
            continue
        costs = (held.recorded_usd, row.recorded_usd)
        merged[key] = replace(
            held,
            tokens=held.tokens + row.tokens,
            recorded_usd=None if None in costs else sum(cost or 0.0 for cost in costs),
        )
    keep = None if primary is None else (primary.model, primary.speed)
    return tuple(replace(row, primary=key == keep) for key, row in merged.items())


@dataclass(frozen=True)
class ProviderUsage:
    """Normalized token counts plus the raw provider mapping.

    ``raw`` keeps the last usage mapping the CLI printed so a caller can
    diagnose a normalization gap without re-parsing the stream.
    ``increment_known=False`` means the invocation's token increment could
    not be reconstructed (for example, a Codex resume without a baseline).
    Its token counts are zero placeholders, with ``turns`` still counted;
    callers must check this marker before pricing or budgeting the counts.
    """

    tokens: TokenUsage = field(default_factory=TokenUsage)
    total_cost_usd: float | None = None
    provider: str = ""
    raw: Mapping[str, Any] | None = None
    increment_known: bool = True
    #: Where ``total_cost_usd`` came from. ``UNKNOWN_MODEL`` and ``NOT_REPORTED`` go with a
    #: ``None`` cost: no price is guessed. Left out with no cost, it is ``NOT_REPORTED``; it is
    #: ``None`` only on a usage built with a cost and no source, which a parser does not do.
    cost_source: CostSource | None = None
    #: Each model the turn ran, including ones ``tokens`` leaves out (subagents, advisors).
    #: Empty when the provider reports no split, or the increment is not known.
    by_model: tuple[ModelUsage, ...] = ()

    def __post_init__(self) -> None:
        """Say ``NOT_REPORTED`` for a usage with no cost and no stated source."""
        if self.cost_source is None and self.total_cost_usd is None:
            object.__setattr__(self, "cost_source", CostSource.NOT_REPORTED)

    def to_dict(self) -> dict[str, Any]:
        """Flatten the normalized counts, cost, and provider name into one mapping.

        ``raw`` is deliberately left out: it is provider-shaped and unbounded,
        while this mapping is meant to be cheap to log on every turn.
        """
        return {
            **self.tokens.to_dict(),
            "total_cost_usd": self.total_cost_usd,
            "provider": self.provider,
            "increment_known": self.increment_known,
            "cost_source": None if self.cost_source is None else self.cost_source.value,
            "by_model": [row.to_dict() for row in self.by_model],
        }
