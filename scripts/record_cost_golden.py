"""Record what agentshim charges for a fixed corpus of recorded usage.

Recorded cost feeds replay and budgets, so it must not move by accident. This
script prices a corpus with the public pricing API and writes the result to
``tests/fixtures/recorded_costs.json``; ``--check`` compares instead of
writing and exits non-zero on any difference, which is what the golden test
does.

Run it without flags only when a price is meant to change, and say so in the
release notes: the diff of the fixture is the list of costs that moved.

The corpus has three parts, each read through public API only:

* every pinned rate, exactly as ``default_pricing()`` holds it;
* a fixed set of token counts priced under every model name a caller might
  use for each pinned model (dated snapshot, bracketed variant, ``vendor/``
  prefix, other case);
* the usage and cost that the Claude and Codex parsers report for the
  recorded streams under ``tests/fixtures``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

from agentshim import TokenUsage, cost_usd, default_pricing, price_for
from agentshim.providers.claude import ClaudeStreamParser
from agentshim.providers.codex import CodexStreamParser

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"
OUTPUT_PATH = FIXTURES / "recorded_costs.json"

#: Seed of the generated token counts; changing it changes the golden.
SEED = 20261030
GENERATED_USAGES = 24

#: Recorded streams replayed through their parser: ``(provider, file name)``.
STREAMS = (
    ("claude", "haiku_task_subagent.jsonl"),
    ("claude", "sonnet_1h_cache.jsonl"),
    ("claude", "haiku45_two_model_calls.jsonl"),
    ("claude", "skill_invoked_haiku.jsonl"),
    ("codex", "exec_gpt6_luna.jsonl"),
    ("codex", "skill_invoked_gpt6_luna.jsonl"),
)

_PROVIDER_OF_VENDOR = {"anthropic": "claude", "openai": "codex"}


def _usages() -> list[TokenUsage]:
    """Edge counts, then seeded random ones that exercise every token class."""
    rng = random.Random(SEED)  # noqa: S311 - a fixed corpus, not a secret
    usages = [
        TokenUsage(),
        TokenUsage(input_tokens=1_000_000),
        TokenUsage(output_tokens=1_000_000, reasoning_output_tokens=1_000_000),
        TokenUsage(
            input_tokens=5_000_000_000,
            cache_read_input_tokens=3_000_000_000,
            cache_write_input_tokens=1_000_000_000,
            cache_write_1h_input_tokens=400_000_000,
            output_tokens=2_000_000_000,
        ),
    ]
    for _ in range(GENERATED_USAGES):
        read = rng.randrange(0, 9_000_000)
        write = rng.randrange(0, 3_000_000)
        output = rng.randrange(0, 2_000_000)
        usages.append(
            TokenUsage(
                input_tokens=read + write + rng.randrange(0, 4_000_000),
                cache_read_input_tokens=read,
                cache_write_input_tokens=write,
                cache_write_1h_input_tokens=rng.randrange(0, write + 1),
                output_tokens=output,
                reasoning_output_tokens=rng.randrange(0, output + 1),
                turns=rng.randrange(0, 40),
            )
        )
    return usages


def _name_forms(provider: str, vendor: str, model: str) -> list[tuple[str, str]]:
    """``(provider, model)`` spellings that must all resolve to the same pinned row."""
    forms = [
        (provider, model),
        (provider, model.upper()),
        (provider, f"  {model} "),
        (vendor, model),
        ("opencode", f"{vendor}/{model}"),
        (provider, f"{model}[1m]"),
    ]
    if provider == "claude":
        forms.append((provider, f"{model}-20251001"))
    return forms


#: The columns of a pinned row that decide a cost; provenance and notes may change freely.
_RATE_COLUMNS = (
    "vendor",
    "model",
    "input_usd",
    "cache_read_usd",
    "cache_write_usd",
    "cache_write_1h_usd",
    "output_usd",
    "reasoning_usd",
)


def _rates() -> list[dict[str, Any]]:
    return [
        {column: row[column] for column in _RATE_COLUMNS}
        for row in (entry.to_dict() for entry in default_pricing().entries)
    ]


def _costs() -> list[dict[str, Any]]:
    usages = _usages()
    rows: list[dict[str, Any]] = []
    for entry in default_pricing().entries:
        provider = _PROVIDER_OF_VENDOR[entry.vendor]
        for name_provider, name in _name_forms(provider, entry.vendor, entry.model):
            pricing = price_for(name_provider, name)
            rows.append(
                {
                    "provider": name_provider,
                    "model": name,
                    "resolved": None if pricing is None else list(pricing.key),
                    "costs": (
                        None if pricing is None else [cost_usd(usage, pricing) for usage in usages]
                    ),
                }
            )
    return rows


def _replay(provider: str, name: str) -> dict[str, Any]:
    parser = ClaudeStreamParser if provider == "claude" else CodexStreamParser
    stream = parser(lambda _event: None)
    for line in (FIXTURES / provider / name).read_text().splitlines(keepends=True):
        stream.feed_stdout(line)
    usage = stream.finish().usage
    return {
        "provider": provider,
        "stream": name,
        "tokens": usage.tokens.to_dict(),
        "total_cost_usd": usage.total_cost_usd,
        "cost_source": None if usage.cost_source is None else usage.cost_source.value,
    }


def build() -> dict[str, Any]:
    """The golden document for the code that is installed now."""
    return {
        "usages": [usage.to_dict() for usage in _usages()],
        "rates": _rates(),
        "costs": _costs(),
        "streams": [_replay(provider, name) for provider, name in STREAMS],
    }


def render(document: dict[str, Any]) -> str:
    """The checked-in text: stable key order, one level of indent."""
    return json.dumps(document, indent=1, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Write the golden, or with ``--check`` report whether it differs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare instead of writing")
    args = parser.parse_args(argv)
    text = render(build())
    if args.check:
        if OUTPUT_PATH.read_text() != text:
            print(f"{OUTPUT_PATH} differs from what the installed code charges", file=sys.stderr)
            return 1
        return 0
    OUTPUT_PATH.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
