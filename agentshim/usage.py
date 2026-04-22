"""Token/turn usage value objects for CLI-based coding agents.

Duplicated in shape (not in import) from ``libs/pydantic_agent/_usage.py``
so ``agentshim`` does not depend on the pydantic-ai stack. The
``to_dict()`` output shape is the shared contract between the two copies
and is pinned by ``tests/unit/test_usage_schema_parity.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TokenUsage:
    """Token counts for a CLI-agent session.

    Invariant: ``cached_input_tokens`` is a subset of ``input_tokens``.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    turns: int = 0

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
            turns=self.turns + other.turns,
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "turns": self.turns,
        }


@dataclass(frozen=True)
class ProviderUsage:
    """Per-session usage bundle: common token counts + provider extras.

    ``provider`` is the registered provider name ("claude", "codex",
    "gemini", "opencode"). ``total_cost_usd`` is ``None`` when the
    provider does not report cost in its stream (codex, gemini, opencode
    depending on config).
    """

    tokens: TokenUsage = field(default_factory=TokenUsage)
    total_cost_usd: float | None = None
    provider: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.tokens.to_dict(),
            "total_cost_usd": self.total_cost_usd,
            "provider": self.provider,
        }
