"""LiteLLM price documents shaped like the real one: partial, renamed, and full of placeholders.

The idealised document in a test lists exactly the models the test asks about, each with a real
price. The real ``model_prices_and_context_window.json`` is several thousand rows: some models
are dropped or renamed between days, a new model can be listed at zero before its price is known,
and many rows are not chat models at all (a rerank row has no per-token price; the ``sample_spec``
row is all zero). These generators lay such a document over the models agentshim prices.
"""

from __future__ import annotations

import json
from typing import Any

from agentshim import Rates
from agentshim.rates import bundled_rates
from hypothesis import strategies as st

PINNED = bundled_rates()
NAMES = tuple(sorted(PINNED.entries))
#: What the generator may do to the row of a pinned model.
FATES = (
    "same",
    "repriced",
    "missing",
    "renamed",
    "zero",
    "zero_output",
    "malformed",
    "zero_fast",
)


def row(
    rates: Rates, *, fast_multiple: float | None = None, zero_priority: bool = False
) -> dict[str, Any]:
    """A chat-model row as LiteLLM writes it: USD per token, with the fields around it."""
    per = 1_000_000
    body: dict[str, Any] = {
        "litellm_provider": "anthropic",
        "mode": "chat",
        "max_input_tokens": 1_000_000,
        "supports_vision": True,
        "input_cost_per_token": rates.input_usd / per,
        "output_cost_per_token": rates.output_usd / per,
        "cache_read_input_token_cost": rates.cache_read_usd / per,
        "cache_creation_input_token_cost": rates.cache_write_usd / per,
    }
    if fast_multiple is not None:
        body["provider_specific_entry"] = {"fast": fast_multiple}
    if zero_priority:
        body["input_cost_per_token_priority"] = 0
        body["output_cost_per_token_priority"] = 0
    return body


def document_bytes(document: dict[str, Any]) -> bytes:
    """The document as a fetch returns it."""
    return json.dumps(document).encode()


def _rows(name: str, fate: str, scale: float) -> dict[str, Any]:
    """What the document holds for the pinned model *name* when *fate* befalls it."""
    standard = PINNED.entries[name].standard
    zero = Rates(0, 0, 0, 0)
    scaled = Rates(
        standard.input_usd * scale,
        standard.output_usd * scale,
        standard.cache_read_usd * scale,
        standard.cache_write_usd * scale,
    )
    rows: dict[str, dict[str, Any]] = {
        "same": {name: row(standard)},
        "repriced": {name: row(scaled)},
        "missing": {},
        "renamed": {f"vendor/{name}-renamed": row(standard)},
        "zero": {name: row(zero)},
        "zero_output": {name: row(Rates(standard.input_usd, 0, 0, 0))},
        "malformed": {
            name: {"mode": "chat", "input_cost_per_token": "free", "output_cost_per_token": None}
        },
        "zero_fast": {name: row(standard, zero_priority=True)},
    }
    return rows[fate]


@st.composite
def fetched_documents(draw: st.DrawFn) -> tuple[dict[str, Any], dict[str, str]]:
    """A document over the pinned models, and what became of each: ``(document, fate by name)``."""
    document: dict[str, Any] = {}
    fates: dict[str, str] = {}
    for name in NAMES:
        fates[name] = draw(st.sampled_from(FATES))
        scale = draw(st.integers(min_value=1, max_value=40)) / 8
        document.update(_rows(name, fates[name], scale))
    others: dict[str, dict[str, Any]] = {
        "sample_spec": row(Rates(0, 0, 0, 0)),
        "a-rerank-model": {"mode": "rerank", "input_cost_per_query": 0.002},
        "a-new-model": row(Rates(0, 0, 0, 0)),
        "a-priced-new-model": row(Rates(1.0, 5.0, 0.1, 1.25)),
    }
    document.update({key: value for key, value in others.items() if draw(st.booleans())})
    return document, fates
