"""Claude output shaped like a steered process, and the model keys and speeds a CLI may print.

``test_by_model.py`` feeds the parser the idealised stream: one ``result`` frame, a model key that is
a real name. These tests hold the invariants over ``stream_shapes.py``, which prints several
results with cumulative ``modelUsage``, and over the odd keys a CLI can print. Each failed on 0.24.0.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from agentshim import CostSource, ModelUsage, ProviderUsage, UsageSpeed, cost_usd, price_for
from agentshim.providers.claude import ClaudeStreamParser
from agentshim.providers.claude.cost import ModelCounts
from hypothesis import given, settings
from hypothesis import strategies as st

from .stream_shapes import PRICED, UNPRICED, SteeredRun, steered_runs, usage_of, wire

if TYPE_CHECKING:
    from collections.abc import Mapping

    from agentshim import AgentEvent


def _parse(
    lines: list[str], baseline: Mapping[str, ModelCounts] | None = None
) -> tuple[ProviderUsage, float | None]:
    events: list[AgentEvent] = []
    parser = ClaudeStreamParser(events.append, model_usage_baseline=baseline)
    for line in lines:
        parser.feed_stdout(line + "\n")
    parsed = parser.finish()
    return parsed.usage, parsed.cost_usd


def _in_order(values: list[float | None]) -> float:
    total = 0.0
    for value in values:
        total += value or 0.0
    return total


def _with_synthetic(line: str, extra: ModelCounts) -> str:
    """The frame with the CLI's placeholder model added to its ``modelUsage``, if it has one."""
    frame = json.loads(line)
    if "modelUsage" in frame:
        frame["modelUsage"] = {"<synthetic>": wire(extra), **frame["modelUsage"]}
    return json.dumps(frame)


class TestASteeredProcess:
    @settings(max_examples=150)
    @given(run=steered_runs())
    def test_results_one_after_another_report_what_one_result_of_the_whole_would(
        self, run: SteeredRun
    ) -> None:
        steered, cost = _parse(run.lines(), run.baseline)
        folded, folded_cost = _parse(run.folded(), run.baseline)
        assert steered.by_model == folded.by_model
        assert steered.tokens == folded.tokens
        assert (steered.cost_source, cost) == (folded.cost_source, folded_cost)

    @settings(max_examples=150)
    @given(run=steered_runs())
    def test_the_rows_of_a_steered_run_add_up_to_its_cost_once(self, run: SteeredRun) -> None:
        usage, cost = _parse(run.lines(), run.baseline)
        assert cost is not None
        assert _in_order([row.recorded_usd for row in usage.by_model]) == cost
        assert sum(row.primary for row in usage.by_model) == (1 if usage.by_model else 0)
        final = run.results[-1].cumulative
        for row in usage.by_model:
            held = run.baseline.get(row.model, ModelCounts())
            assert row.tokens.output_tokens == final[row.model].output_tokens - held.output_tokens

    def test_the_first_message_is_not_counted_again_by_the_second(self) -> None:
        # Result 1 reports 1000 in and 500 out; result 2 reports the process's cumulative 2000 and 1000.
        model = "claude-opus-5-5"
        first = ModelCounts(input_tokens=1000, output_tokens=500)
        second = ModelCounts(input_tokens=2000, output_tokens=1000)
        lines = [
            json.dumps(
                {
                    "type": "result",
                    "subtype": "success",
                    "result": "ok",
                    "num_turns": 1,
                    "usage": usage_of(counts, 0, "standard"),
                    "modelUsage": {model: wire(total)},
                }
            )
            for counts, total in (
                (first, first),
                (ModelCounts(input_tokens=1000, output_tokens=500), second),
            )
        ]
        usage, cost = _parse(lines)
        (row,) = usage.by_model
        assert (row.tokens.input_tokens, row.tokens.output_tokens) == (2000, 1000)
        assert usage.tokens.output_tokens == 1000
        assert row.recorded_usd == cost == pytest.approx(0.028)


class TestWhatACliMayPrint:
    @settings(max_examples=150)
    @given(
        named=st.dictionaries(
            st.sampled_from([*PRICED, UNPRICED]), st.integers(0, 10_000), max_size=3
        ),
        blank=st.dictionaries(
            st.sampled_from(["", " ", "\t", "  \n"]), st.integers(0, 10_000), max_size=2
        ),
    )
    def test_a_model_key_with_no_name_does_not_fail_the_turn(
        self, named: dict[str, int], blank: dict[str, int]
    ) -> None:
        keys = {**{k: ModelCounts(output_tokens=v) for k, v in named.items()}}
        keys.update({k: ModelCounts(output_tokens=v) for k, v in blank.items()})
        line = json.dumps(
            {
                "type": "result",
                "subtype": "success",
                "result": "ok",
                "usage": {"input_tokens": 1, "output_tokens": 1},
                "modelUsage": {k: wire(v) for k, v in keys.items()},
            }
        )
        usage, cost = _parse([line])  # must not raise
        assert all(row.model.strip() for row in usage.by_model)
        unnamed = sum(v for v in blank.values())
        if unnamed:
            assert (cost, usage.cost_source) == (None, CostSource.UNKNOWN_MODEL)
            (row,) = [row for row in usage.by_model if row.model == "unknown"]
            assert row.tokens.output_tokens == unnamed
            assert row.recorded_usd is None

    @settings(max_examples=100)
    @given(run=steered_runs(), tokens=st.integers(min_value=1, max_value=10**6))
    def test_the_synthetic_placeholder_bills_nothing_and_changes_nothing(
        self, run: SteeredRun, tokens: int
    ) -> None:
        extra = ModelCounts(input_tokens=tokens, output_tokens=tokens)
        lines = [_with_synthetic(line, extra) for line in run.lines()]
        assert _parse(lines, run.baseline) == _parse(run.lines(), run.baseline)

    def test_an_ultrafast_result_makes_its_primary_row_ultrafast(self) -> None:
        counts = ModelCounts(output_tokens=5)
        line = json.dumps(
            {
                "type": "result",
                "subtype": "success",
                "result": "ok",
                "usage": usage_of(counts, 0, "ultrafast"),
                "modelUsage": {"claude-opus-5-5": wire(counts)},
            }
        )
        (row,) = _parse([line])[0].by_model
        assert row.speed is UsageSpeed.ULTRAFAST


class TestRecordedUsdIsTheStandardTierListPrice:
    def test_the_contract_is_stated_where_a_caller_reads_it(self) -> None:
        doc = ModelUsage.__doc__ or ""
        assert "standard\n            tier" in doc or "standard tier" in doc
        assert "not a bill" in doc
        assert "speed_premium_usd" in doc
        guide = (Path(__file__).resolve().parents[2] / "docs" / "pricing.md").read_text()
        assert "## Usage by model" in guide
        assert "### Billing a fast turn" in guide
        assert "**standard tier**" in guide

    # Passes on 0.24.0 too: the behavior was always this. It is pinned here because the field name
    # read like a bill and the number was never one.
    @pytest.mark.parametrize("speed", ["standard", "fast"])
    def test_a_fast_row_is_priced_like_a_standard_one_and_sums_to_the_turn_cost(
        self, speed: str
    ) -> None:
        counts = ModelCounts(input_tokens=1_000_000, output_tokens=1_000_000)
        line = json.dumps(
            {
                "type": "result",
                "subtype": "success",
                "result": "ok",
                "usage": usage_of(counts, 0, speed),
                "modelUsage": {"claude-opus-5-5": wire(counts)},
            }
        )
        usage, cost = _parse([line])
        (row,) = usage.by_model
        pricing = price_for("claude", "claude-opus-5-5")
        assert pricing is not None
        assert row.tokens.input_tokens == 1_000_000
        assert row.recorded_usd == cost == usage.total_cost_usd == cost_usd(row.tokens, pricing)
