"""Per-turn token usage out of Codex's cumulative thread totals.

``thread/tokenUsage/updated`` reports the *thread's* running total, so a turn's
usage is the difference between the total when the turn ended and the total
when the previous reported turn ended. The totals are kept in ``ProviderUsage.raw``
under the same keys the one-shot path uses (``input_tokens``,
``cached_input_tokens``, ...), so a checkpoint written by either path seeds the
other. A resumed conversation with no baseline cannot tell where its turn
began, so its first turn reports ``increment_known=False`` instead of a guess.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agentshim.core.usage import ProviderUsage, TokenUsage, normalized_usage
from agentshim.providers.codex.provider import PROFILE

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .protocol import TokenUsageBreakdown

#: A thread's running totals, keyed like Codex's ``exec`` usage object.
Totals = dict[str, int]

_KEYS = (
    "input_tokens",
    "cached_input_tokens",
    "cache_write_input_tokens",
    "output_tokens",
    "reasoning_output_tokens",
    "total_tokens",
)


def zero_totals() -> Totals:
    """The totals of a thread that has run nothing."""
    return dict.fromkeys(_KEYS, 0)


def totals_of(breakdown: TokenUsageBreakdown) -> Totals:
    """Express a wire breakdown as running totals."""
    return {
        "input_tokens": breakdown.input_tokens,
        "cached_input_tokens": breakdown.cached_input_tokens,
        "cache_write_input_tokens": breakdown.cache_write_input_tokens,
        "output_tokens": breakdown.output_tokens,
        "reasoning_output_tokens": breakdown.reasoning_output_tokens,
        "total_tokens": breakdown.total_tokens,
    }


def baseline_from(previous: ProviderUsage | None) -> Totals | None:
    """The totals a resumed conversation continues from, or ``None`` if unknown."""
    if previous is None or previous.provider != PROFILE.name or previous.raw is None:
        return None
    return _read(previous.raw)


def _read(raw: Mapping[str, object]) -> Totals:
    totals = zero_totals()
    for key in _KEYS:
        value = raw.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            totals[key] = value
    return totals


def turn_usage(total: Totals | None, baseline: Totals | None) -> ProviderUsage:
    """Usage of the turn that moved the thread from *baseline* to *total*.

    ``increment_known`` is false when either is missing; the tokens are then
    zero placeholders and only ``turns`` is counted, as in the one-shot path.
    The cost is the caller's to add: it depends on the model.
    """
    if total is None or baseline is None:
        return ProviderUsage(
            tokens=TokenUsage(turns=1),
            provider=PROFILE.name,
            raw=None if total is None else dict(total),
            increment_known=False,
        )
    tokens = normalized_usage(
        input_tokens=total["input_tokens"] - baseline["input_tokens"],
        output_tokens=total["output_tokens"] - baseline["output_tokens"],
        cache_read_input_tokens=total["cached_input_tokens"] - baseline["cached_input_tokens"],
        cache_write_input_tokens=(
            total["cache_write_input_tokens"] - baseline["cache_write_input_tokens"]
        ),
        reasoning_output_tokens=(
            total["reasoning_output_tokens"] - baseline["reasoning_output_tokens"]
        ),
        turns=1,
    )
    return ProviderUsage(
        tokens=tokens,
        provider=PROFILE.name,
        raw=dict(total),
    )
