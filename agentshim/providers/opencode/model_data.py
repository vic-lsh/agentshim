"""opencode-owned model facts."""

import datetime as dt

from agentshim.core.models import ModelSpec

GOOGLE_MODELS_URL = "https://ai.google.dev/gemini-api/docs/models"
_MODEL_CHECKED = dt.date(2026, 10, 17)


def _google(model: str, context_window: int, display_name: str, tier: tuple[int, str]) -> ModelSpec:
    return ModelSpec(
        vendor="google",
        model=model,
        context_window=context_window,
        source=f"{GOOGLE_MODELS_URL}/{model}",
        checked=_MODEL_CHECKED,
        display_name=display_name,
        family="Gemini",
        tier=tier[0],
        tier_name=tier[1],
        generation_id="3",
        generation="Gemini 3",
        generation_rank=1,
    )


_PRO = (1, "Pro")
_GOOGLE_1M = 1_048_576

MODEL_ENTRIES = (_google("gemini-3-pro-preview", _GOOGLE_1M, "Gemini 3 Pro Preview", _PRO),)
