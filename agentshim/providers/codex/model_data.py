"""Codex-owned model and price facts."""

import datetime as dt
from dataclasses import replace

from agentshim.core.models import ModelSpec
from agentshim.core.pricing import ModelPricing

OPENAI_MODELS_URL = "https://developers.openai.com/api/docs/models"
OPENAI_PRICING_URL = "https://developers.openai.com/api/docs/pricing"
_MODEL_CHECKED = dt.date(2026, 10, 17)
_PRICE_CHECKED = dt.date(2026, 9, 27)
#: ``gpt-6.1-sol`` was read from the page on its own date, after the rest of the table.
_SOL_6_1_CHECKED = dt.date(2026, 10, 30)


def _openai_model(
    model: str,
    context_window: int,
    display_name: str,
    version: str,
    tier: tuple[int, str] | None = None,
) -> ModelSpec:
    generation_id, generation_rank = _GENERATIONS[version]
    return ModelSpec(
        vendor="openai",
        model=model,
        context_window=context_window,
        source=f"{OPENAI_MODELS_URL}/{model}",
        checked=_MODEL_CHECKED,
        display_name=display_name,
        family="GPT",
        tier=None if tier is None else tier[0],
        tier_name=None if tier is None else tier[1],
        generation_id=generation_id,
        generation=f"GPT-{version}",
        generation_rank=generation_rank,
    )


#: GPT's generations by version, newest first: ``(generation_id, rank)``.
_GENERATIONS = {
    version: (version, rank) for rank, version in enumerate(("6.1", "6", "5.6", "5.5"), start=1)
}

_LUNA = (1, "Luna")
_TERRA = (2, "Terra")
_SOL = (3, "Sol")
_ASTRA = (4, "Astra")
_OPENAI_1_05M = 1_050_000

MODEL_ENTRIES = (
    _openai_model("gpt-6-astra", _OPENAI_1_05M, "GPT-6 Astra", "6", _ASTRA),
    _openai_model("gpt-6.1-sol", _OPENAI_1_05M, "GPT-6.1 Sol", "6.1", _SOL),
    _openai_model("gpt-6-sol", _OPENAI_1_05M, "GPT-6 Sol", "6", _SOL),
    _openai_model("gpt-6-luna", _OPENAI_1_05M, "GPT-6 Luna", "6", _LUNA),
    _openai_model("gpt-5.6-sol", _OPENAI_1_05M, "GPT-5.6 Sol", "5.6", _SOL),
    _openai_model("gpt-5.6-terra", _OPENAI_1_05M, "GPT-5.6 Terra", "5.6", _TERRA),
    _openai_model("gpt-5.6-luna", _OPENAI_1_05M, "GPT-5.6 Luna", "5.6", _LUNA),
    _openai_model("gpt-5.5", _OPENAI_1_05M, "GPT-5.5", "5.5"),
)

_OPENAI_NOTE = (
    "Short-context standard tier; the long-context rates on the same page are not modelled."
)
_OPENAI_NO_WRITE_NOTE = (
    "Short-context standard tier; the page lists no cache-write rate, so cache writes "
    "are priced at base input."
)


def _openai(
    model: str,
    input_usd: float,
    cache_read_usd: float,
    output_usd: float,
    cache_write_usd: float | None = None,
) -> ModelPricing:
    return ModelPricing(
        vendor="openai",
        model=model,
        input_usd=input_usd,
        cache_read_usd=cache_read_usd,
        cache_write_usd=input_usd if cache_write_usd is None else cache_write_usd,
        output_usd=output_usd,
        source=OPENAI_PRICING_URL,
        checked=_PRICE_CHECKED,
        note=_OPENAI_NOTE if cache_write_usd is not None else _OPENAI_NO_WRITE_NOTE,
    )


PRICING_ENTRIES = (
    _openai("gpt-6-astra", 10.00, 1.00, 50.00, cache_write_usd=12.50),
    replace(
        _openai("gpt-6.1-sol", 2.00, 0.10, 10.00, cache_write_usd=2.50),
        checked=_SOL_6_1_CHECKED,
    ),
    _openai("gpt-6-sol", 2.00, 0.20, 10.00, cache_write_usd=2.50),
    _openai("gpt-6-luna", 0.10, 0.01, 0.50, cache_write_usd=0.125),
    _openai("gpt-5.6-sol", 4.00, 0.40, 20.00, cache_write_usd=5.00),
    _openai("gpt-5.6-terra", 2.00, 0.20, 12.00, cache_write_usd=2.50),
    _openai("gpt-5.6-luna", 0.20, 0.02, 1.20, cache_write_usd=0.25),
    _openai("gpt-5.5", 5.00, 0.50, 30.00),
    _openai("gpt-5.4", 2.50, 0.25, 15.00),
    _openai("gpt-5.4-mini", 0.75, 0.075, 4.50),
    _openai("gpt-5.4-nano", 0.20, 0.02, 1.25),
    _openai("gpt-5.3-codex", 1.75, 0.175, 14.00),
    _openai("gpt-5.2", 1.75, 0.175, 14.00),
    _openai("gpt-5.1", 1.25, 0.125, 10.00),
    _openai("gpt-5", 1.25, 0.125, 10.00),
    _openai("gpt-5-mini", 0.25, 0.025, 2.00),
    _openai("gpt-5-nano", 0.05, 0.005, 0.40),
)
