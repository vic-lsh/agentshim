"""Token and cost accounting shared by every provider."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True)
class TokenUsage:
    """Token counts for one turn.

    Invariant on every provider: ``cached_input_tokens <= input_tokens``.
    Providers that report cache tokens disjoint from input tokens (Claude)
    fold them in before constructing this.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_input_tokens: int = 0
    reasoning_output_tokens: int = 0
    turns: int = 0

    def __add__(self, other: TokenUsage) -> TokenUsage:
        """Accumulate a running total over the turns of a session.

        Every field is summed, ``turns`` included, so folding per-turn usages
        together yields both the token totals and the turn count.
        """
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
            cache_write_input_tokens=self.cache_write_input_tokens + other.cache_write_input_tokens,
            reasoning_output_tokens=self.reasoning_output_tokens + other.reasoning_output_tokens,
            turns=self.turns + other.turns,
        )

    def to_dict(self) -> dict[str, int]:
        """Flatten the counts into a JSON-serializable mapping keyed by field name."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "cache_write_input_tokens": self.cache_write_input_tokens,
            "reasoning_output_tokens": self.reasoning_output_tokens,
            "turns": self.turns,
        }


@dataclass(frozen=True)
class ProviderUsage:
    """Normalized token counts plus the raw provider mapping.

    ``raw`` keeps the last usage mapping the CLI printed so a caller can
    diagnose a normalization gap without re-parsing the stream.
    """

    tokens: TokenUsage = field(default_factory=TokenUsage)
    total_cost_usd: float | None = None
    provider: str = ""
    raw: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Flatten the normalized counts, cost, and provider name into one mapping.

        ``raw`` is deliberately left out: it is provider-shaped and unbounded,
        while this mapping is meant to be cheap to log on every turn.
        """
        return {
            **self.tokens.to_dict(),
            "total_cost_usd": self.total_cost_usd,
            "provider": self.provider,
        }
