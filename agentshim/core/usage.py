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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

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
    :mod:`agentshim.core.pricing`.
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
        }
