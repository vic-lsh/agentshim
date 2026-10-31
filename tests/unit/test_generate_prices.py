"""The pinned prices are generated from a checked-in LiteLLM snapshot, and nothing else."""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from agentshim import default_pricing, parse_litellm, price_for
from agentshim.providers.claude import prices_generated as claude_prices
from agentshim.providers.codex import model_data as codex_data

from tests.unit.test_pricing import UNPRICED_BY_DESIGN

if TYPE_CHECKING:
    from types import ModuleType

REPO = Path(__file__).resolve().parents[2]
GENERATOR_PATH = REPO / "scripts" / "generate_prices.py"


@pytest.fixture(scope="module")
def generator() -> ModuleType:
    """The generator script, imported by path (``scripts/`` is not a package)."""
    name = "generate_prices"
    spec = importlib.util.spec_from_file_location(name, GENERATOR_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolve annotations through sys.modules
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


def test_regenerating_from_the_snapshot_reproduces_every_checked_in_file(
    generator: ModuleType,
) -> None:
    snapshot = generator.read_snapshot()
    files = {generator.SNAPSHOT_PATH: generator.render_snapshot(snapshot)}
    files.update(generator.outputs(snapshot))
    for path, text in files.items():
        assert path.read_text() == text, f"{path.relative_to(REPO)} was edited by hand"


def test_the_snapshot_holds_exactly_the_configured_models_and_says_where_it_is_from(
    generator: ModuleType,
) -> None:
    snapshot = generator.read_snapshot()
    configured = [model for vendor in generator.VENDORS for model in vendor.models]
    assert sorted(snapshot["models"]) == sorted(configured)
    assert dt.date.fromisoformat(snapshot["fetched_on"])
    assert len(snapshot["upstream_sha256"]) == 64
    assert snapshot["source"].startswith("https://")
    assert str(REPO) not in json.dumps(snapshot)


def test_a_model_the_catalogue_excludes_is_never_priced(generator: ModuleType) -> None:
    configured = {model for vendor in generator.VENDORS for model in vendor.models}
    assert configured.isdisjoint(UNPRICED_BY_DESIGN)
    for model in UNPRICED_BY_DESIGN:
        assert price_for("opencode", f"google-vertex/{model}") is None


def test_a_configured_model_the_source_lacks_stops_the_release(generator: ModuleType) -> None:
    full = {
        model: {"input_cost_per_token": 1e-6, "output_cost_per_token": 2e-6}
        for vendor in generator.VENDORS
        for model in vendor.models
    }
    del full["gpt-5"]
    with pytest.raises(generator.GeneratorError, match="gpt-5"):
        generator.trim(full)


def test_trimming_keeps_the_columns_the_parser_reads_and_drops_the_rest(
    generator: ModuleType,
) -> None:
    row = {
        "input_cost_per_token": 1e-6,
        "output_cost_per_token": 2e-6,
        "input_cost_per_token_priority": 3e-6,
        "cache_creation_input_token_cost_above_1hr": 4e-6,
        "provider_specific_entry": {"fast": 2},
        "input_cost_per_token_batches": 5e-7,
        "input_cost_per_token_above_272k_tokens": 9e-6,
        "supports_vision": True,
    }
    full = {model: dict(row) for vendor in generator.VENDORS for model in vendor.models}
    kept = generator.trim(full)["gpt-5"]
    assert set(kept) == set(row) - {
        "input_cost_per_token_batches",
        "input_cost_per_token_above_272k_tokens",
        "supports_vision",
    }


def test_the_snapshot_date_dates_the_recorded_table_and_names_its_version(
    generator: ModuleType,
) -> None:
    day = dt.date.fromisoformat(generator.read_snapshot()["fetched_on"])
    table = default_pricing()
    assert table.version == f"litellm-{day.isoformat()}"
    assert table.last_updated == day
    assert all(entry.checked == day for entry in table.entries)


def test_each_pinned_row_is_what_the_library_parser_reads_from_the_snapshot(
    generator: ModuleType,
) -> None:
    snapshot = generator.read_snapshot()
    day = dt.date.fromisoformat(snapshot["fetched_on"])
    parsed = parse_litellm(json.dumps(snapshot["models"]).encode(), checked_on=day)
    pinned = {entry.model: entry for entry in default_pricing().entries}
    assert set(pinned) == set(parsed.entries)
    for model, entry in pinned.items():
        rates = parsed.entries[model].standard
        assert (
            entry.input_usd,
            entry.cache_read_usd,
            entry.cache_write_usd,
            entry.cache_write_1h_usd,
            entry.output_usd,
        ) == (
            rates.input_usd,
            rates.cache_read_usd,
            rates.cache_write_usd,
            rates.cache_write_1h_usd,
            rates.output_usd,
        )
        assert entry.reasoning_usd is None


def test_a_model_without_a_cache_write_rate_is_priced_at_input_and_says_so() -> None:
    inferred = codex_data.INFERRED_CACHE_WRITE
    assert inferred
    for entry in codex_data.PRICING_ENTRIES:
        if entry.model in inferred:
            assert entry.cache_write_usd == entry.input_usd
            assert "priced at base input" in entry.note
        else:
            assert "priced at base input" not in entry.note
    assert not claude_prices.INFERRED_CACHE_WRITE
