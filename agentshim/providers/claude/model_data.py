"""Claude-owned model and price facts."""

import datetime as dt

from agentshim.core.models import ModelSpec
from agentshim.core.pricing import ModelPricing
from agentshim.core.rates import RateEntry, RateTable

from .prices_generated import RATE_ENTRIES, SNAPSHOT_DATE

ANTHROPIC_CONTEXT_WINDOWS_URL = (
    "https://platform.claude.com/docs/en/build-with-claude/context-windows"
)
ANTHROPIC_PRICING_URL = "https://platform.claude.com/docs/en/about-claude/pricing"
_MODEL_CHECKED = dt.date(2026, 10, 17)


def _anthropic_model(
    model: str, context_window: int, tier: tuple[int, str], version: str
) -> ModelSpec:
    generation_id, generation_rank = _GENERATIONS[version]
    return ModelSpec(
        vendor="anthropic",
        model=model,
        context_window=context_window,
        source=ANTHROPIC_CONTEXT_WINDOWS_URL,
        checked=_MODEL_CHECKED,
        display_name=f"{tier[1]} {version}",
        family="Claude",
        tier=tier[0],
        tier_name=tier[1],
        generation_id=generation_id,
        generation=f"Claude {version}",
        generation_rank=generation_rank,
    )


#: Claude's generations by version, newest first: ``(generation_id, rank)``.
_GENERATIONS = {
    version: (version, rank)
    for rank, version in enumerate(("5.5", "5.1", "5", "4.8", "4.7", "4.6", "4.5"), start=1)
}

_HAIKU = (1, "Haiku")
_SONNET = (2, "Sonnet")
_OPUS = (3, "Opus")
_FABLE = (4, "Fable")
_ONE_MILLION = 1_000_000

MODEL_ENTRIES = (
    _anthropic_model("claude-fable-5-1", _ONE_MILLION, _FABLE, "5.1"),
    _anthropic_model("claude-fable-5", _ONE_MILLION, _FABLE, "5"),
    _anthropic_model("claude-opus-5-5", _ONE_MILLION, _OPUS, "5.5"),
    _anthropic_model("claude-opus-5", _ONE_MILLION, _OPUS, "5"),
    _anthropic_model("claude-opus-4-8", _ONE_MILLION, _OPUS, "4.8"),
    _anthropic_model("claude-opus-4-7", _ONE_MILLION, _OPUS, "4.7"),
    _anthropic_model("claude-opus-4-6", _ONE_MILLION, _OPUS, "4.6"),
    _anthropic_model("claude-sonnet-5-5", _ONE_MILLION, _SONNET, "5.5"),
    _anthropic_model("claude-sonnet-5", _ONE_MILLION, _SONNET, "5"),
    _anthropic_model("claude-sonnet-4-6", _ONE_MILLION, _SONNET, "4.6"),
    _anthropic_model("claude-haiku-5-5", _ONE_MILLION, _HAIKU, "5.5"),
    _anthropic_model("claude-haiku-4-5", 200_000, _HAIKU, "4.5"),
)


_NOTE = "Claude API first-party, global inference; thinking is billed as output."
_NOTES = {
    "claude-haiku-5-5": (
        "Prices for prompts up to 100,000 tokens; a request whose prompt is over "
        "100,000 tokens pays 0.50 input, 0.625 cache write, 1.00 1h cache write, "
        "0.05 cache read and 2.50 output, which this table leaves out."
    ),
}


def _anthropic(entry: RateEntry) -> ModelPricing:
    rates = entry.standard
    return ModelPricing(
        vendor="anthropic",
        model=entry.name,
        input_usd=rates.input_usd,
        cache_read_usd=rates.cache_read_usd,
        cache_write_usd=rates.cache_write_usd,
        cache_write_1h_usd=rates.cache_write_1h_usd,
        output_usd=rates.output_usd,
        source=ANTHROPIC_PRICING_URL,
        checked=SNAPSHOT_DATE,
        note=f"{_NOTE} {_NOTES.get(entry.name, '')}".strip(),
    )


PRICING_ENTRIES = tuple(_anthropic(entry) for entry in RATE_ENTRIES)
#: Flat rates with speed tiers, for estimates; the same generated numbers as ``PRICING_ENTRIES``.
RATES = RateTable(SNAPSHOT_DATE, {entry.name: entry for entry in RATE_ENTRIES})

#: The ``model`` of a message the CLI wrote itself rather than a model call, which bills nothing.
SYNTHETIC_MODEL = "<synthetic>"
#: Names the CLI reports that are not a model with a price: its placeholder for a message that
#: never reached the API, and the bare family names, which do not say which generation ran.
UNPRICEABLE_MODELS = frozenset({SYNTHETIC_MODEL, "synthetic", "opus", "sonnet", "haiku", "fable"})
