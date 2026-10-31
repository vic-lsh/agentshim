"""Codex-owned model and price facts."""

import datetime as dt

from agentshim.core.models import ModelSpec
from agentshim.core.pricing import ModelPricing
from agentshim.core.rates import RateEntry, RateTable

from .prices_generated import INFERRED_CACHE_WRITE, RATE_ENTRIES, SNAPSHOT_DATE

OPENAI_MODELS_URL = "https://developers.openai.com/api/docs/models"
OPENAI_PRICING_URL = "https://developers.openai.com/api/docs/pricing"
_MODEL_CHECKED = dt.date(2026, 10, 17)


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


def _openai(entry: RateEntry) -> ModelPricing:
    rates = entry.standard
    return ModelPricing(
        vendor="openai",
        model=entry.name,
        input_usd=rates.input_usd,
        cache_read_usd=rates.cache_read_usd,
        cache_write_usd=rates.cache_write_usd,
        output_usd=rates.output_usd,
        source=OPENAI_PRICING_URL,
        checked=SNAPSHOT_DATE,
        note=_OPENAI_NO_WRITE_NOTE if entry.name in INFERRED_CACHE_WRITE else _OPENAI_NOTE,
    )


PRICING_ENTRIES = tuple(_openai(entry) for entry in RATE_ENTRIES)
#: Flat rates with speed tiers, for estimates; the same generated numbers as ``PRICING_ENTRIES``.
RATES = RateTable(SNAPSHOT_DATE, {entry.name: entry for entry in RATE_ENTRIES})
