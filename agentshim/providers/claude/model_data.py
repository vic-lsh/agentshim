"""Claude-owned model and price facts."""

import datetime as dt

from agentshim.core.models import ModelSpec
from agentshim.core.pricing import ModelPricing

ANTHROPIC_CONTEXT_WINDOWS_URL = (
    "https://platform.claude.com/docs/en/build-with-claude/context-windows"
)
ANTHROPIC_PRICING_URL = "https://platform.claude.com/docs/en/about-claude/pricing"
_MODEL_CHECKED = dt.date(2026, 10, 17)
_PRICE_CHECKED = dt.date(2026, 10, 19)


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


def _anthropic(  # noqa: PLR0913 - one positional per price column
    model: str,
    input_usd: float,
    cache_write_usd: float,
    cache_write_1h_usd: float,
    cache_read_usd: float,
    output_usd: float,
    note: str = "",
) -> ModelPricing:
    return ModelPricing(
        vendor="anthropic",
        model=model,
        input_usd=input_usd,
        cache_read_usd=cache_read_usd,
        cache_write_usd=cache_write_usd,
        cache_write_1h_usd=cache_write_1h_usd,
        output_usd=output_usd,
        source=ANTHROPIC_PRICING_URL,
        checked=_PRICE_CHECKED,
        note=f"{_NOTE} {note}".strip(),
    )


PRICING_ENTRIES = (
    _anthropic("claude-fable-5-1", 10.00, 12.50, 20.00, 0.25, 50.00),
    _anthropic("claude-fable-5", 10.00, 12.50, 20.00, 1.00, 50.00),
    _anthropic("claude-opus-5-5", 4.00, 5.00, 8.00, 0.20, 20.00),
    _anthropic("claude-opus-5", 5.00, 6.25, 10.00, 0.50, 25.00),
    _anthropic("claude-opus-4-8", 5.00, 6.25, 10.00, 0.50, 25.00),
    _anthropic("claude-opus-4-7", 5.00, 6.25, 10.00, 0.50, 25.00),
    _anthropic("claude-opus-4-6", 5.00, 6.25, 10.00, 0.50, 25.00),
    _anthropic("claude-sonnet-5-5", 2.00, 2.50, 4.00, 0.10, 10.00),
    _anthropic("claude-sonnet-5", 2.00, 2.50, 4.00, 0.20, 10.00),
    _anthropic("claude-sonnet-4-6", 3.00, 3.75, 6.00, 0.30, 15.00),
    _anthropic(
        "claude-haiku-5-5",
        0.10,
        0.125,
        0.20,
        0.01,
        0.50,
        note=(
            "Prices for prompts up to 100,000 tokens; a request whose prompt is over "
            "100,000 tokens pays 0.50 input, 0.625 cache write, 1.00 1h cache write, "
            "0.05 cache read and 2.50 output, which this table leaves out."
        ),
    ),
    _anthropic("claude-haiku-4-5", 1.00, 1.25, 2.00, 0.10, 5.00),
)
