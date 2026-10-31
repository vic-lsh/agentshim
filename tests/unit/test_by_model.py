"""Usage by model: the split a turn's totals leave out, and the price each part was recorded at."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest
from agentshim import (
    CostSource,
    ModelUsage,
    ProviderUsage,
    TokenUsage,
    UsageSpeed,
    merge_model_usage,
    price_for,
)
from agentshim.providers.claude import ClaudeStreamParser
from agentshim.providers.claude.cost import ModelCounts, mark_primary, priced_turn
from agentshim.providers.codex.speed import speed_of_tier
from agentshim.testing import CodexScript, Spend
from hypothesis import given
from hypothesis import strategies as st

from tests.unit.providers.codex.app_server.harness import rig

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from agentshim import AgentEvent

KNOWN = ("claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-4-5")
UNKNOWN = "claude-imaginary-9"
_COUNT = st.integers(min_value=0, max_value=2_000_000)


def _counts(draw: st.DrawFn) -> ModelCounts:
    return ModelCounts(
        input_tokens=draw(_COUNT),
        output_tokens=draw(_COUNT),
        cache_read_tokens=draw(_COUNT),
        cache_write_tokens=draw(_COUNT),
        reasoning_tokens=draw(st.integers(min_value=0, max_value=2_000_000)),
    )


@st.composite
def model_usages(draw: st.DrawFn, *, models: tuple[str, ...] = KNOWN) -> dict[str, ModelCounts]:
    chosen = draw(st.lists(st.sampled_from(models), min_size=1, max_size=len(models), unique=True))
    return {model: _counts(draw) for model in chosen}


def _wire(counts: ModelCounts) -> dict[str, int]:
    return {
        "inputTokens": counts.input_tokens,
        "outputTokens": counts.output_tokens,
        "cacheReadInputTokens": counts.cache_read_tokens,
        "cacheCreationInputTokens": counts.cache_write_tokens,
        "thinkingTokens": counts.reasoning_tokens,
    }


def _stream(
    models: Mapping[str, ModelCounts],
    *,
    init: str | None,
    main: str | None,
    speed: str | None = "standard",
    num_turns: int = 3,
) -> list[str]:
    usage: dict[str, Any] = {
        "input_tokens": 1,
        "output_tokens": 2,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }
    if speed is not None:
        usage["speed"] = speed
    lines: list[dict[str, Any]] = []
    if init is not None:
        lines.append({"type": "system", "subtype": "init", "session_id": "s", "model": init})
    if main is not None:
        lines.append(
            {
                "type": "assistant",
                "message": {"id": "m1", "model": main, "content": [{"type": "text", "text": "hi"}]},
            }
        )
    lines.append(
        {
            "type": "result",
            "subtype": "success",
            "result": "ok",
            "num_turns": num_turns,
            "usage": usage,
            "modelUsage": {model: _wire(counts) for model, counts in models.items()},
        }
    )
    return [json.dumps(line) + "\n" for line in lines]


def _in_order(values: Iterable[float | None]) -> float:
    """Left to right, as recorded cost is added; ``sum`` may compensate and differ in the last bit."""
    total = 0.0
    for value in values:
        total += value or 0.0
    return total


def _parse(lines: list[str]) -> tuple[ProviderUsage, float | None]:
    events: list[AgentEvent] = []
    parser = ClaudeStreamParser(events.append)
    for line in lines:
        parser.feed_stdout(line)
    parsed = parser.finish()
    return parsed.usage, parsed.cost_usd


class TestClaudeSplit:
    def test_a_subagent_model_has_its_own_row_the_main_totals_leave_out(self) -> None:
        models = {
            "claude-opus-5-5": ModelCounts(input_tokens=10, output_tokens=500),
            "claude-haiku-4-5": ModelCounts(input_tokens=7_000, output_tokens=900),
        }
        usage, cost = _parse(_stream(models, init="claude-opus-5-5", main="claude-opus-5-5"))
        assert [row.model for row in usage.by_model] == ["claude-opus-5-5", "claude-haiku-4-5"]
        assert usage.tokens.output_tokens == 2  # the main conversation only
        assert sum(row.tokens.output_tokens for row in usage.by_model) == 1400
        assert cost == _in_order(row.recorded_usd for row in usage.by_model)

    def test_the_model_the_conversation_ran_on_is_primary_and_carries_speed_and_turns(
        self,
    ) -> None:
        models = {
            "claude-haiku-4-5": ModelCounts(input_tokens=1, output_tokens=9_000),
            "claude-opus-5-5": ModelCounts(input_tokens=1, output_tokens=5),
        }
        usage, _ = _parse(
            _stream(
                models, init="claude-opus-5-5", main="claude-opus-5-5", speed="fast", num_turns=4
            )
        )
        haiku, opus = usage.by_model
        assert (opus.primary, opus.speed, opus.tokens.turns) == (True, UsageSpeed.FAST, 4)
        assert (haiku.primary, haiku.speed, haiku.tokens.turns) == (False, None, 0)

    def test_a_dated_snapshot_and_a_bracketed_variant_name_the_same_model(self) -> None:
        models = {
            "claude-haiku-4-5-20251001": ModelCounts(output_tokens=9_000),
            "claude-opus-5-5[1m]": ModelCounts(output_tokens=5),
        }
        usage, _ = _parse(_stream(models, init="claude-opus-5-5", main="claude-opus-5-5"))
        assert [row.primary for row in usage.by_model] == [False, True]

    def test_without_a_launch_model_the_model_with_the_most_output_is_primary(self) -> None:
        models = {
            "claude-haiku-4-5": ModelCounts(output_tokens=100),
            "claude-opus-5-5": ModelCounts(output_tokens=100),
            "claude-sonnet-5-5": ModelCounts(output_tokens=300),
        }
        usage, _ = _parse(_stream(models, init=None, main=None))
        assert [row.primary for row in usage.by_model] == [False, False, True]

    def test_a_tie_goes_to_the_first_model_listed(self) -> None:
        models = {
            "claude-haiku-4-5": ModelCounts(output_tokens=100),
            "claude-opus-5-5": ModelCounts(output_tokens=100),
        }
        usage, _ = _parse(_stream(models, init=None, main=None))
        assert [row.primary for row in usage.by_model] == [True, False]

    def test_a_speed_the_result_does_not_report_stays_unknown(self) -> None:
        models = {"claude-opus-5-5": ModelCounts(output_tokens=1)}
        usage, _ = _parse(_stream(models, init="claude-opus-5-5", main=None, speed=None))
        assert usage.by_model[0].speed is None

    def test_a_synthetic_message_does_not_choose_the_primary(self) -> None:
        models = {
            "claude-haiku-4-5": ModelCounts(output_tokens=900),
            "claude-opus-5-5": ModelCounts(output_tokens=5),
        }
        usage, _ = _parse(_stream(models, init="claude-opus-5-5", main="<synthetic>"))
        assert [row.primary for row in usage.by_model] == [False, True]

    def test_thinking_tokens_fill_reasoning_inside_output(self) -> None:
        models = {"claude-opus-5-5": ModelCounts(output_tokens=100, reasoning_tokens=60)}
        usage, _ = _parse(_stream(models, init="claude-opus-5-5", main=None))
        assert usage.by_model[0].tokens.reasoning_output_tokens == 60
        assert usage.by_model[0].tokens.output_tokens == 100

    def test_an_unpriced_model_has_no_price_and_the_priced_ones_keep_theirs(self) -> None:
        models = {
            "claude-opus-5-5": ModelCounts(input_tokens=1_000_000),
            UNKNOWN: ModelCounts(input_tokens=1_000_000),
        }
        usage, cost = _parse(_stream(models, init="claude-opus-5-5", main=None))
        opus, unknown = usage.by_model
        assert cost is None
        assert usage.cost_source is CostSource.UNKNOWN_MODEL
        assert unknown.recorded_usd is None
        assert opus.recorded_usd == pytest.approx(4.0)

    def test_a_result_without_model_usage_has_no_split(self) -> None:
        line = json.dumps(
            {"type": "result", "subtype": "success", "result": "ok", "total_cost_usd": 1.5}
        )
        usage, _ = _parse([line + "\n"])
        assert usage.by_model == ()
        assert usage.cost_source is CostSource.PROVIDER_REPORTED

    def test_a_followon_turn_inside_one_run_adds_into_the_same_rows(self) -> None:
        first = _stream(
            {"claude-opus-5-5": ModelCounts(input_tokens=10, output_tokens=100)},
            init="claude-opus-5-5",
            main="claude-opus-5-5",
            num_turns=2,
        )
        second = _stream(
            {
                "claude-opus-5-5": ModelCounts(input_tokens=10, output_tokens=100),
                "claude-haiku-4-5": ModelCounts(output_tokens=50),
            },
            init=None,
            main="claude-opus-5-5",
            num_turns=3,
        )
        # The second result's modelUsage is cumulative, so give the parser the first as baseline.
        events: list[AgentEvent] = []
        parser = ClaudeStreamParser(events.append)
        for line in (*first, *second[1:]):
            parser.feed_stdout(line)
        usage = parser.finish().usage
        opus = next(row for row in usage.by_model if row.model == "claude-opus-5-5")
        assert opus.tokens.output_tokens == 200
        assert opus.tokens.turns == 5
        assert sum(row.primary for row in usage.by_model) == 1


@given(models=model_usages(models=(*KNOWN, UNKNOWN)), data=st.data())
def test_every_turn_has_one_primary_and_priced_rows_sum_to_the_total(
    models: dict[str, ModelCounts], data: st.DataObject
) -> None:
    first = data.draw(st.sampled_from(list(models)))
    speed = data.draw(st.sampled_from(["standard", "fast", None]))
    turns = data.draw(st.integers(min_value=0, max_value=50))
    lines = _stream(models, init=first, main=None, speed=speed, num_turns=turns)
    usage, cost = _parse(lines)
    live = [model for model, counts in models.items() if not counts.empty]
    assert [row.model for row in usage.by_model] == live
    assert sum(row.primary for row in usage.by_model) == (1 if live else 0)
    for row in usage.by_model:
        assert (row.speed is not None) == row.primary or (speed is None)
    assert sum(row.tokens.turns for row in usage.by_model) == (turns if live else 0)
    if all(price_for("claude", model) is not None for model in live):
        assert cost is not None
        assert _in_order(row.recorded_usd for row in usage.by_model) == cost
    else:
        assert cost is None
        priced = [row for row in usage.by_model if price_for("claude", row.model) is not None]
        assert all(row.recorded_usd is not None for row in priced)
        assert all(row.recorded_usd is None for row in usage.by_model if row not in priced)


@given(models=model_usages(), baseline=model_usages())
def test_a_turn_split_by_a_baseline_prices_only_what_accrued(
    models: dict[str, ModelCounts], baseline: dict[str, ModelCounts]
) -> None:
    turn = priced_turn(models, baseline, one_hour_share=0.0)
    for row in turn.by_model:
        accrued = models[row.model].since(baseline.get(row.model))
        assert row.tokens.output_tokens == accrued.output_tokens
        assert not accrued.empty


def test_a_model_without_tokens_gets_no_row() -> None:
    turn = priced_turn({"claude-opus-5-5": ModelCounts()}, {}, one_hour_share=0.0)
    assert turn.by_model == ()
    assert turn.usd == 0.0


@given(
    st.lists(
        st.tuples(st.sampled_from(KNOWN), st.sampled_from([None, UsageSpeed.FAST]), _COUNT),
        max_size=6,
    ),
    st.lists(
        st.tuples(st.sampled_from(KNOWN), st.sampled_from([None, UsageSpeed.FAST]), _COUNT),
        max_size=6,
    ),
)
def test_merging_two_turns_keeps_every_token_and_one_primary(
    a: list[tuple[str, UsageSpeed | None, int]], b: list[tuple[str, UsageSpeed | None, int]]
) -> None:
    def rows(spec: list[tuple[str, UsageSpeed | None, int]]) -> tuple[ModelUsage, ...]:
        seen: set[tuple[str, UsageSpeed | None]] = set()
        out: list[ModelUsage] = []
        for model, speed, tokens in spec:
            if (model, speed) in seen:
                continue
            seen.add((model, speed))
            out.append(
                ModelUsage(model, TokenUsage(output_tokens=tokens), recorded_usd=1.0, speed=speed)
            )
        return mark_primary(out, [None], speed=None, turns=0) if out else ()

    first, second = rows(a), rows(b)
    merged = merge_model_usage(first, second)
    assert sum(row.tokens.output_tokens for row in merged) == sum(
        row.tokens.output_tokens for row in (*first, *second)
    )
    assert sum(row.recorded_usd or 0.0 for row in merged) == pytest.approx(
        sum(row.recorded_usd or 0.0 for row in (*first, *second))
    )
    assert sum(row.primary for row in merged) == (1 if merged else 0)


def test_a_merged_sum_is_unpriced_if_either_part_was() -> None:
    priced = ModelUsage("m", TokenUsage(output_tokens=1), recorded_usd=2.0, primary=True)
    unpriced = ModelUsage("m", TokenUsage(output_tokens=1), recorded_usd=None)
    assert merge_model_usage([priced], [unpriced])[0].recorded_usd is None


def test_a_row_rejects_a_negative_or_non_finite_cost_and_an_empty_model() -> None:
    with pytest.raises(ValueError, match="recorded_usd"):
        ModelUsage("m", TokenUsage(), recorded_usd=-1.0)
    with pytest.raises(ValueError, match="recorded_usd"):
        ModelUsage("m", TokenUsage(), recorded_usd=float("nan"))
    with pytest.raises(ValueError, match="model"):
        ModelUsage(" ", TokenUsage())


def test_to_dict_only_adds_a_by_model_key() -> None:
    row = ModelUsage("m", TokenUsage(output_tokens=3), 0.5, UsageSpeed.FAST, primary=True)
    flat = ProviderUsage(provider="x", by_model=(row,)).to_dict()
    assert flat["by_model"] == [row.to_dict()]
    assert flat["by_model"][0]["speed"] == "fast"
    assert ProviderUsage(provider="x").to_dict()["by_model"] == []
    json.dumps(flat)


class TestCodex:
    def test_a_turn_has_one_primary_row_at_the_model_the_server_started_the_thread_with(
        self,
    ) -> None:
        script = CodexScript(model="gpt-6-luna")
        script.turn(Spend(input_tokens=1_000_000, output_tokens=1_000_000))
        r = rig(script)
        conversation = r.open(model="gpt-6-luna")
        usage = r.turn(conversation).usage
        conversation.close()
        (row,) = usage.by_model
        assert (row.model, row.primary, row.speed) == ("gpt-6-luna", True, UsageSpeed.STANDARD)
        assert row.tokens == usage.tokens
        assert row.recorded_usd == usage.total_cost_usd == pytest.approx(0.10 + 0.50)

    def test_the_servers_service_tier_sets_the_speed(self) -> None:
        script = CodexScript(model="gpt-6-luna", service_tier="priority")
        script.turn(Spend(input_tokens=10, output_tokens=1))
        r = rig(script)
        conversation = r.open(model="gpt-6-luna")
        (row,) = r.turn(conversation).usage.by_model
        conversation.close()
        assert row.speed is UsageSpeed.FAST

    def test_an_unpriced_model_still_has_a_row_with_no_price(self) -> None:
        script = CodexScript()
        script.turn(Spend(input_tokens=10, output_tokens=1))
        r = rig(script)
        conversation = r.open()
        usage = r.turn(conversation).usage
        conversation.close()
        (row,) = usage.by_model
        assert row.model == "fake-model"
        assert row.recorded_usd is None
        assert usage.total_cost_usd is None

    def test_a_resume_with_no_baseline_has_no_split_to_report(self) -> None:
        script = CodexScript()
        script.add_thread("old")
        script.turn(Spend(input_tokens=300, output_tokens=30))
        r = rig(script)
        conversation = r.open(resume_id="old")
        usage = r.turn(conversation).usage
        conversation.close()
        assert not usage.increment_known
        assert usage.by_model == ()

    @pytest.mark.parametrize(
        ("tier", "speed"),
        [
            (None, UsageSpeed.STANDARD),
            ("default", UsageSpeed.STANDARD),
            ("flex", UsageSpeed.STANDARD),
            ("priority", UsageSpeed.FAST),
            ("fast", UsageSpeed.FAST),
            ("ultrafast", UsageSpeed.ULTRAFAST),
        ],
    )
    def test_service_tiers_map_to_speeds(self, tier: str | None, speed: UsageSpeed) -> None:
        assert speed_of_tier(tier) is speed
