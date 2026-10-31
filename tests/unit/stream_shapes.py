"""Claude Code ``stream-json`` output shaped like a process that answered several messages.

The idealised stream in a test is one ``result`` frame. A Claude process that is steered, or that
is kept alive across turns, prints one ``result`` per message, and its ``modelUsage`` is
cumulative over the process while its ``usage`` and ``num_turns`` are each message's own. A model
(a subagent's) can first show up in a later result. These generators make that shape.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from agentshim.providers.claude.cost import ModelCounts
from hypothesis import strategies as st

PRICED = ("claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-4-5")
UNPRICED = "claude-imaginary-9"
_TOKENS = st.integers(min_value=0, max_value=300_000)


def wire(counts: ModelCounts) -> dict[str, int]:
    """One model's counts as ``modelUsage`` prints them."""
    return {
        "inputTokens": counts.input_tokens,
        "outputTokens": counts.output_tokens,
        "cacheReadInputTokens": counts.cache_read_tokens,
        "cacheCreationInputTokens": counts.cache_write_tokens,
        "thinkingTokens": counts.reasoning_tokens,
    }


def accumulated(a: ModelCounts, b: ModelCounts) -> ModelCounts:
    """Counts after a process spent *b* more: what its next ``result`` reports for the model."""
    return ModelCounts(
        input_tokens=a.input_tokens + b.input_tokens,
        output_tokens=a.output_tokens + b.output_tokens,
        cache_read_tokens=a.cache_read_tokens + b.cache_read_tokens,
        cache_write_tokens=a.cache_write_tokens + b.cache_write_tokens,
        reasoning_tokens=a.reasoning_tokens + b.reasoning_tokens,
    )


@dataclass(frozen=True)
class Result:
    """One ``result`` frame."""

    usage: dict[str, Any]  # this message's own tokens, the main conversation only
    turns: int
    cumulative: dict[str, ModelCounts]  # modelUsage: everything the process has spent

    def line(self) -> str:
        """The frame as the CLI prints it."""
        return json.dumps(
            {
                "type": "result",
                "subtype": "success",
                "result": "ok",
                "num_turns": self.turns,
                "usage": self.usage,
                "modelUsage": {model: wire(counts) for model, counts in self.cumulative.items()},
            }
        )


def usage_of(counts: ModelCounts, one_hour: int, speed: str | None) -> dict[str, Any]:
    """A message's ``usage`` mapping for the tokens of its main conversation."""
    body: dict[str, Any] = {
        "input_tokens": counts.input_tokens,
        "output_tokens": counts.output_tokens,
        "cache_read_input_tokens": counts.cache_read_tokens,
        "cache_creation_input_tokens": counts.cache_write_tokens,
        "cache_creation": {"ephemeral_1h_input_tokens": min(one_hour, counts.cache_write_tokens)},
    }
    if speed is not None:
        body["speed"] = speed
    return body


@dataclass(frozen=True)
class SteeredRun:
    """A process that printed several results, and what it had spent before the first."""

    main: str
    results: tuple[Result, ...]
    baseline: dict[str, ModelCounts]
    speed: str | None

    def lines(self) -> list[str]:
        """The stream: the model it started with, then each result."""
        init = json.dumps(
            {"type": "system", "subtype": "init", "session_id": "s", "model": self.main}
        )
        return [init, *(result.line() for result in self.results)]

    def folded(self) -> list[str]:
        """The same run as one result: the totals, with the cumulative counts at its end."""
        total = {
            key: sum(result.usage.get(key, 0) for result in self.results)
            for key in (
                "input_tokens",
                "output_tokens",
                "cache_read_input_tokens",
                "cache_creation_input_tokens",
            )
        }
        one_hour = sum(
            result.usage["cache_creation"]["ephemeral_1h_input_tokens"] for result in self.results
        )
        usage: dict[str, Any] = {**total, "cache_creation": {"ephemeral_1h_input_tokens": one_hour}}
        if self.speed is not None:
            usage["speed"] = self.speed
        last = self.results[-1]
        merged = Result(usage, sum(r.turns for r in self.results), last.cumulative)
        init = json.dumps(
            {"type": "system", "subtype": "init", "session_id": "s", "model": self.main}
        )
        return [init, merged.line()]


@st.composite
def steered_runs(draw: st.DrawFn, *, models: tuple[str, ...] = PRICED) -> SteeredRun:
    """1 to 4 results over 1 to 3 models, a later one possibly first used in a later result."""
    chosen = draw(st.lists(st.sampled_from(models), min_size=1, max_size=3, unique=True))
    speed = draw(st.sampled_from(["standard", "fast", None]))
    baseline = {
        model: ModelCounts(input_tokens=draw(_TOKENS), output_tokens=draw(_TOKENS))
        for model in chosen
        if draw(st.booleans())
    }
    running = dict(baseline)
    results = []
    for _ in range(draw(st.integers(min_value=1, max_value=4))):
        step = {
            model: ModelCounts(
                input_tokens=draw(_TOKENS),
                output_tokens=draw(_TOKENS),
                cache_read_tokens=draw(_TOKENS),
                cache_write_tokens=draw(_TOKENS),
            )
            for model in chosen
            if draw(st.booleans())
        }
        for model, counts in step.items():
            running[model] = accumulated(running[model], counts) if model in running else counts
        main_step = step.get(chosen[0], ModelCounts())
        results.append(
            Result(
                usage_of(main_step, draw(_TOKENS), speed),
                draw(st.integers(min_value=1, max_value=5)),
                dict(running),
            )
        )
    return SteeredRun(chosen[0], tuple(results), baseline, speed)
