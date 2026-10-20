"""A Claude turn is priced from ``modelUsage`` and agentshim's table, not the CLI's cost."""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING, Any

import pytest
from agentshim import CostSource, UsageReport, price_for
from agentshim.core.events import AgentEvent
from agentshim.core.pricing import cost_usd
from agentshim.core.usage import TokenUsage
from agentshim.providers.claude import ClaudeStreamParser
from hypothesis import given
from hypothesis import strategies as st

if TYPE_CHECKING:
    from collections.abc import Mapping

    from agentshim.providers.claude.cost import ModelCounts

_KNOWN = "claude-sonnet-5-5"
_COUNT = st.integers(min_value=0, max_value=5_000_000)
_COUNTS = st.tuples(_COUNT, _COUNT, _COUNT, _COUNT)


Counts = tuple[int, int, int, int]  # input, output, cache read, cache write


def _model_usage(counts: Counts, canonical: str | None = None) -> dict[str, int | str]:
    inp, out, read, write = counts
    usage: dict[str, int | str] = {
        "inputTokens": inp,
        "outputTokens": out,
        "cacheReadInputTokens": read,
        "cacheCreationInputTokens": write,
    }
    if canonical is not None:
        usage["canonicalModel"] = canonical
    return usage


def _result(
    models: dict[str, dict[str, Any]] | None,
    *,
    cli_cost: float | None,
    one_hour: int = 0,
    write: int = 0,
) -> str:
    payload: dict[str, Any] = {
        "type": "result",
        "subtype": "success",
        "result": "ok",
        "usage": {
            "input_tokens": 1,
            "cache_creation_input_tokens": write,
            "cache_creation": {
                "ephemeral_1h_input_tokens": one_hour,
                "ephemeral_5m_input_tokens": write - one_hour,
            },
        },
    }
    if cli_cost is not None:
        payload["total_cost_usd"] = cli_cost
    if models is not None:
        payload["modelUsage"] = models
    return json.dumps(payload) + "\n"


def _run(
    line: str,
    baseline: Mapping[str, ModelCounts] | None = None,
    cost_baseline: float = 0.0,
) -> tuple[ClaudeStreamParser, list[AgentEvent]]:
    events: list[AgentEvent] = []
    parser = ClaudeStreamParser(
        events.append, cost_baseline_usd=cost_baseline, model_usage_baseline=baseline
    )
    parser.feed_stdout(line)
    return parser, events


def _expected(model: str, counts: Counts, one_hour: int = 0) -> float:
    inp, out, read, write = counts
    pricing = price_for("claude", model)
    assert pricing is not None
    return cost_usd(
        TokenUsage(
            input_tokens=inp + read + write,
            output_tokens=out,
            cache_read_input_tokens=read,
            cache_write_input_tokens=write,
            cache_write_1h_input_tokens=one_hour,
        ),
        pricing,
    )


def test_the_probe_an_old_cli_overstates_sonnet_5_5_by_2_6x() -> None:
    """Claude Code 2.1.268 priced the unknown ``claude-sonnet-5-5`` like Opus 5."""
    models = {_KNOWN: _model_usage((6, 2000, 19_860, 6000), canonical=_KNOWN)}
    parser, events = _run(_result(models, cli_cost=0.11996, one_hour=6000, write=6000))
    parsed = parser.finish()
    assert parsed.cost_usd == pytest.approx(0.045998)
    assert parsed.usage.total_cost_usd == parsed.cost_usd
    assert parsed.usage.cost_source is CostSource.PRICE_TABLE
    reports = [e for e in events if isinstance(e, UsageReport)]
    assert reports[-1].cost_usd == parsed.cost_usd
    assert reports[-1].usage.cost_source is CostSource.PRICE_TABLE


@given(counts=_COUNTS, cli=st.floats(0, 1000), data=st.data())
def test_a_known_model_is_priced_from_its_tokens_whatever_the_cli_said(
    counts: Counts, cli: float, data: st.DataObject
) -> None:
    write = counts[3]
    one_hour = data.draw(st.integers(0, write))
    models = {_KNOWN: _model_usage(counts)}
    parser, _ = _run(_result(models, cli_cost=cli, one_hour=one_hour, write=write))
    parsed = parser.finish()
    assert parsed.usage.cost_source is CostSource.PRICE_TABLE
    assert parsed.cost_usd is not None
    # Each count is rounded to whole tokens, so the 1h share of a write may differ by one.
    low = _expected(_KNOWN, counts, max(one_hour - 1, 0))
    high = _expected(_KNOWN, counts, min(one_hour + 1, write))
    assert min(low, high) - 1e-9 <= parsed.cost_usd <= max(low, high) + 1e-9


@given(first=_COUNTS, more=_COUNTS)
def test_cumulative_model_usage_prices_the_turn_by_its_difference(
    first: Counts, more: Counts
) -> None:
    """Splitting a run into turns never changes what it costs in total."""
    total = (first[0] + more[0], first[1] + more[1], first[2] + more[2], first[3] + more[3])
    one, _ = _run(_result({_KNOWN: _model_usage(first)}, cli_cost=9.0))
    two, _ = _run(
        _result({_KNOWN: _model_usage(total)}, cli_cost=99.0), baseline=one.model_usage_total
    )
    whole, _ = _run(_result({_KNOWN: _model_usage(total)}, cli_cost=99.0))
    first_cost, second_cost, whole_cost = (
        one.finish().cost_usd,
        two.finish().cost_usd,
        whole.finish().cost_usd,
    )
    assert first_cost is not None
    assert second_cost is not None
    assert whole_cost is not None
    assert math.isclose(first_cost + second_cost, whole_cost, rel_tol=1e-9, abs_tol=1e-9)


@given(counts=_COUNTS, cli=st.floats(0, 1000))
def test_an_unknown_model_has_no_cost_and_says_so(counts: Counts, cli: float) -> None:
    models = {_KNOWN: _model_usage(counts), "claude-future-9-9": _model_usage((1, 1, 0, 0))}
    parser, events = _run(_result(models, cli_cost=cli))
    parsed = parser.finish()
    assert parsed.cost_usd is None
    assert parsed.usage.total_cost_usd is None
    assert parsed.usage.cost_source is CostSource.UNKNOWN_MODEL
    reports = [e for e in events if isinstance(e, UsageReport)]
    assert reports[-1].cost_usd is None


def test_an_unknown_id_with_a_known_canonical_model_is_priced_as_the_canonical_one() -> None:
    counts = (100, 100, 0, 0)
    models = {"vendor-alias": _model_usage(counts, canonical=_KNOWN)}
    parsed = _run(_result(models, cli_cost=5.0))[0].finish()
    assert parsed.usage.cost_source is CostSource.PRICE_TABLE
    assert parsed.cost_usd == pytest.approx(_expected(_KNOWN, counts))


def test_an_unknown_model_that_used_no_tokens_this_turn_does_not_unprice_the_turn() -> None:
    idle = {"claude-future-9-9": _model_usage((5, 5, 0, 0))}
    first, _ = _run(_result(idle, cli_cost=1.0))
    models = {**idle, _KNOWN: _model_usage((10, 10, 0, 0))}
    parser, _ = _run(_result(models, cli_cost=2.0), baseline=first.model_usage_total)
    parsed = parser.finish()
    assert parsed.usage.cost_source is CostSource.PRICE_TABLE
    assert parsed.cost_usd == pytest.approx(_expected(_KNOWN, (10, 10, 0, 0)))


def test_a_result_that_names_no_model_keeps_the_clis_cost_and_labels_it() -> None:
    parsed = _run(_result(None, cli_cost=0.5), cost_baseline=0.25)[0].finish()
    assert parsed.cost_usd == 0.25
    assert parsed.usage.cost_source is CostSource.PROVIDER_REPORTED


def test_a_result_with_no_cost_at_all_has_no_source() -> None:
    parsed = _run(_result(None, cli_cost=None))[0].finish()
    assert parsed.cost_usd is None
    assert parsed.usage.cost_source is None


def test_the_source_is_in_the_usage_mapping() -> None:
    models = {_KNOWN: _model_usage((1, 1, 0, 0))}
    parsed = _run(_result(models, cli_cost=1.0))[0].finish()
    assert parsed.usage.to_dict()["cost_source"] == "price_table"
