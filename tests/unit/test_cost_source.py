"""A parsed turn always says where its cost came from, or that it has none.

A reader that finds ``cost_source`` unset must be reading a usage built before the field existed:
a provider that reports a cost, one that reports none, and a model with no price each have their
own answer. These tests hold that over every provider's recorded stream and the Codex transport,
so a new provider that leaves it out fails here, not in a caller that reads ``None`` as "old".
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from agentshim import CostSource, ProviderUsage, TokenUsage
from agentshim.providers.claude import ClaudeStreamParser
from agentshim.providers.codex import CodexStreamParser
from agentshim.providers.copilot import CopilotStreamParser
from agentshim.providers.gemini import GeminiStreamParser
from agentshim.providers.opencode import OpencodeStreamParser
from agentshim.testing import CodexScript, Spend
from hypothesis import given
from hypothesis import strategies as st

from tests.unit.providers.codex.app_server.harness import rig

if TYPE_CHECKING:
    from agentshim.core.events import AgentEvent
    from agentshim.core.provider import ParsedTurn

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
PARSERS = {
    "claude": ClaudeStreamParser,
    "codex": CodexStreamParser,
    "copilot": CopilotStreamParser,
    "gemini": GeminiStreamParser,
    "opencode": OpencodeStreamParser,
}
#: Every recorded stream of every provider the library parses, one parametrization each.
STREAMS = sorted(
    (provider, path.name) for provider in PARSERS for path in (FIXTURES / provider).glob("*.jsonl")
)


def _replay(provider: str, name: str) -> ParsedTurn:
    events: list[AgentEvent] = []
    parser = PARSERS[provider](events.append)
    for line in (FIXTURES / provider / name).read_text().splitlines(keepends=True):
        parser.feed_stdout(line)
    return parser.finish()


@pytest.mark.parametrize(("provider", "name"), STREAMS)
def test_every_recorded_turn_names_where_its_cost_came_from_or_that_it_has_none(
    provider: str, name: str
) -> None:
    usage = _replay(provider, name).usage

    assert usage.cost_source is not None
    # A turn with a cost has a source that makes one; one without has a source that explains it.
    no_cost = {CostSource.UNKNOWN_MODEL, CostSource.NOT_REPORTED}
    assert (usage.total_cost_usd is None) == (usage.cost_source in no_cost)


def test_a_cost_opencode_reports_is_the_providers_own() -> None:
    usage = _replay("opencode", "step_finish.jsonl").usage

    assert usage.total_cost_usd is not None
    assert usage.cost_source is CostSource.PROVIDER_REPORTED


@pytest.mark.parametrize(
    ("provider", "name"),
    [("codex", "exec_gpt6_luna.jsonl"), ("gemini", "result_stats.jsonl")],
)
def test_a_cli_that_prints_no_cost_says_it_reported_none(provider: str, name: str) -> None:
    usage = _replay(provider, name).usage

    assert usage.total_cost_usd is None
    assert usage.cost_source is CostSource.NOT_REPORTED


def test_a_codex_turn_on_a_model_with_no_price_has_no_cost_and_says_why() -> None:
    script = CodexScript()
    script.turn(Spend(input_tokens=10, output_tokens=1))
    r = rig(script)
    conversation = r.open()
    usage = r.turn(conversation).usage
    conversation.close()

    assert usage.total_cost_usd is None
    assert usage.cost_source is CostSource.UNKNOWN_MODEL
    assert [row.recorded_usd for row in usage.by_model] == [None]


def test_a_codex_turn_on_a_priced_model_is_priced_from_the_table() -> None:
    script = CodexScript(model="gpt-6-luna")
    script.turn(Spend(input_tokens=10, output_tokens=1))
    r = rig(script)
    conversation = r.open(model="gpt-6-luna")
    usage = r.turn(conversation).usage
    conversation.close()

    assert usage.total_cost_usd is not None
    assert usage.cost_source is CostSource.PRICE_TABLE


@given(
    cost=st.none() | st.floats(0, 1e6, allow_nan=False, allow_infinity=False),
    source=st.none() | st.sampled_from(CostSource),
)
def test_a_usage_built_with_no_cost_and_no_source_says_not_reported(
    cost: float | None, source: CostSource | None
) -> None:
    usage = ProviderUsage(tokens=TokenUsage(), total_cost_usd=cost, cost_source=source)

    if source is not None:
        assert usage.cost_source is source
    elif cost is None:
        assert usage.cost_source is CostSource.NOT_REPORTED
    else:
        # A cost with no stated source stays unstated: agentshim does not guess it.
        assert usage.cost_source is None


def test_the_source_is_in_the_usage_mapping_when_it_is_not_reported() -> None:
    assert ProviderUsage().to_dict()["cost_source"] == "not_reported"
