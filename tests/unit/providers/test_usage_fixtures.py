"""Normalized token counts from recorded provider streams.

Each fixture under ``tests/fixtures/<provider>/`` is real CLI stdout, trimmed
(long strings shortened, local paths replaced) with every usage object kept
verbatim. The expected numbers are read off the raw frames by hand, so these
tests pin each provider's mapping onto agentshim's breakdown:

- Codex ``turn.completed``: ``input_tokens`` already includes cache reads
  (``cached_input_tokens``) and writes; ``output_tokens`` already includes
  ``reasoning_output_tokens``.
- Claude ``result.usage``: ``input_tokens`` excludes both cache classes, which
  are reported disjoint; ``cache_creation`` splits writes by TTL.
- opencode ``step_finish``: ``input`` excludes cache reads and writes;
  ``output`` excludes ``reasoning``.
- Gemini ``result.stats``: ``input_tokens`` includes ``cached``.

Copilot recordings are covered in ``copilot/test_fixtures.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from agentshim import TokenUsage, cost_usd, price_for
from agentshim.providers.claude import ClaudeStreamParser
from agentshim.providers.codex import CodexStreamParser
from agentshim.providers.gemini import GeminiStreamParser
from agentshim.providers.opencode import OpencodeStreamParser

if TYPE_CHECKING:
    from agentshim.core.events import AgentEvent
    from agentshim.core.provider import ParsedTurn

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
PARSERS = {
    "claude": ClaudeStreamParser,
    "codex": CodexStreamParser,
    "gemini": GeminiStreamParser,
    "opencode": OpencodeStreamParser,
}


def replay(provider: str, name: str) -> ParsedTurn:
    events: list[AgentEvent] = []
    parser = PARSERS[provider](events.append)
    for line in (FIXTURES / provider / name).read_text().splitlines(keepends=True):
        parser.feed_stdout(line)
    return parser.finish()


@pytest.mark.parametrize(
    ("provider", "name", "expected"),
    [
        pytest.param(
            "codex",
            "exec_gpt6_luna.jsonl",
            TokenUsage(
                input_tokens=288_616,
                cache_read_input_tokens=255_744,
                output_tokens=1_820,
                reasoning_output_tokens=682,
                turns=1,
            ),
            id="codex",
        ),
        pytest.param(
            "claude",
            "haiku_task_subagent.jsonl",
            TokenUsage(
                input_tokens=8 + 10_885 + 37_798,
                cache_read_input_tokens=37_798,
                cache_write_input_tokens=10_885,
                output_tokens=578,
                turns=2,
            ),
            id="claude-5m-writes",
        ),
        pytest.param(
            "claude",
            "sonnet_1h_cache.jsonl",
            TokenUsage(
                input_tokens=99 + 172_386 + 6_831_549,
                cache_read_input_tokens=6_831_549,
                cache_write_input_tokens=172_386,
                cache_write_1h_input_tokens=172_386,
                output_tokens=90_873,
                turns=131,
            ),
            id="claude-1h-writes",
        ),
        pytest.param(
            "opencode",
            "step_finish.jsonl",
            TokenUsage(
                input_tokens=20_526 + 49_841,
                cache_read_input_tokens=49_841,
                output_tokens=223 + 705,
                reasoning_output_tokens=705,
                turns=5,
            ),
            id="opencode",
        ),
        pytest.param(
            "gemini",
            "result_stats.jsonl",
            TokenUsage(
                input_tokens=193_576,
                cache_read_input_tokens=151_333,
                output_tokens=7_732,
                turns=1,
            ),
            id="gemini",
        ),
    ],
)
def test_recorded_usage_normalizes(provider: str, name: str, expected: TokenUsage) -> None:
    parsed = replay(provider, name)
    assert parsed.error is None
    assert parsed.usage.tokens == expected


def test_codex_uncached_input_excludes_cache_reads() -> None:
    tokens = replay("codex", "exec_gpt6_luna.jsonl").usage.tokens
    assert tokens.uncached_input_tokens == 288_616 - 255_744


def test_claude_table_prices_reproduce_the_reported_cost() -> None:
    """Haiku 4.5's table entry reproduces Claude Code's own ``total_cost_usd``.

    The run's ``modelUsage`` covers every request, the Task subagent's
    included, and its writes are all five-minute writes (the result's
    ``cache_creation`` has no one-hour part), so pricing its counts must give
    the reported cost exactly.
    """
    pricing = price_for("claude", "claude-haiku-4-5-20251001")
    assert pricing is not None
    every_request = TokenUsage(
        input_tokens=15_400 + 805_904 + 71_750,
        cache_read_input_tokens=805_904,
        cache_write_input_tokens=71_750,
        output_tokens=8_150,
    )
    parsed = replay("claude", "haiku_task_subagent.jsonl")
    assert parsed.cost_usd == pytest.approx(0.2264279)
    assert cost_usd(every_request, pricing) == pytest.approx(parsed.cost_usd, rel=1e-9)


def test_claude_result_usage_leaves_out_subagent_requests() -> None:
    """Known gap: ``result.usage`` is the main conversation only.

    Pricing it undercounts a run that delegated to a Task subagent, while the
    CLI's ``total_cost_usd`` covers every request.
    """
    parsed = replay("claude", "haiku_task_subagent.jsonl")
    pricing = price_for("claude", "claude-haiku-4-5")
    assert pricing is not None
    assert parsed.cost_usd is not None
    assert cost_usd(parsed.usage.tokens, pricing) < parsed.cost_usd
