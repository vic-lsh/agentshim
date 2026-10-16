"""The static model table, and the models each provider profile declares."""

from __future__ import annotations

import datetime as dt
import re
from typing import Any

import pytest
from agentshim import (
    ModelSpec,
    ModelTable,
    context_window,
    default_models,
    get_provider,
    model_spec,
    provider_names,
)
from hypothesis import assume, given
from hypothesis import strategies as st

TABLE = default_models()
CHECKED = dt.date(2026, 10, 16)
KNOWN_IDS = {entry.model for entry in TABLE.entries}
_DATE_SNAPSHOT = re.compile(r"-\d{8}$")


def _spec(**overrides: object) -> ModelSpec:
    fields: dict[str, Any] = {
        "vendor": "openai",
        "model": "test-model",
        "context_window": 1000,
        "source": "https://example.com/models",
        "checked": CHECKED,
    }
    fields.update(overrides)
    return ModelSpec(**fields)


class TestShippedTable:
    def test_it_is_versioned_and_dated(self) -> None:
        assert TABLE.version.strip()
        assert isinstance(TABLE.last_updated, dt.date)

    @pytest.mark.parametrize(
        "entry", TABLE.entries, ids=lambda entry: f"{entry.vendor}/{entry.model}"
    )
    def test_every_entry_names_its_source_and_check_date(self, entry: ModelSpec) -> None:
        assert entry.source.startswith("https://")
        assert entry.checked <= TABLE.last_updated
        assert entry.context_window > 0

    @pytest.mark.parametrize(
        ("provider", "model", "window"),
        [
            ("claude", "claude-fable-5-1", 1_000_000),
            ("claude", "claude-opus-5-5", 1_000_000),
            ("claude", "claude-sonnet-5-5", 1_000_000),
            ("claude", "claude-haiku-5-5", 1_000_000),
            ("claude", "claude-opus-5", 1_000_000),
            ("claude", "claude-haiku-4-5", 200_000),
            ("codex", "gpt-6-luna", 1_050_000),
            ("codex", "gpt-5.6-sol", 1_050_000),
        ],
    )
    def test_current_models_have_their_published_window(
        self, provider: str, model: str, window: int
    ) -> None:
        assert context_window(provider, model) == window


class TestLookup:
    def test_a_vendor_prefix_and_a_date_snapshot_resolve_to_the_model(self) -> None:
        assert context_window("opencode", "anthropic/claude-opus-5-5") == 1_000_000
        assert context_window("opencode", "openai/gpt-5.5") == 1_050_000
        assert context_window("claude", "claude-haiku-4-5-20251001") == 200_000

    def test_a_bracketed_variant_is_unknown(self) -> None:
        assert context_window("claude", "claude-opus-5-5[1m]") is None

    def test_a_missing_model_or_a_provider_with_no_vendor_is_unknown(self) -> None:
        assert context_window("claude", None) is None
        assert context_window("claude", "") is None
        assert context_window("gemini", "gemini-3-pro-preview") is None
        assert context_window("opencode", "google-vertex/gemini-3-pro-preview") is None

    def test_a_model_run_through_another_vendors_provider_is_unknown(self) -> None:
        assert context_window("codex", "claude-opus-5-5") is None

    @given(
        known=st.sampled_from(sorted(KNOWN_IDS)),
        suffix=st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789.-", min_size=1),
    )
    def test_a_known_id_with_anything_appended_is_unknown(self, known: str, suffix: str) -> None:
        """The table never matches by prefix: a longer name is another model."""
        name = known + suffix
        assume(name not in KNOWN_IDS and not _DATE_SNAPSHOT.search(name))
        assert context_window("claude", name) is None
        assert context_window("codex", name) is None

    @given(st.text(max_size=40))
    def test_a_name_the_table_lacks_is_unknown(self, name: str) -> None:
        base = _DATE_SNAPSHOT.sub("", name.strip().lower().rpartition("/")[2])
        assume(base not in KNOWN_IDS)
        for provider in provider_names():
            assert model_spec(provider, name) is None

    def test_a_caller_table_overrides_the_shipped_one(self) -> None:
        table = TABLE.with_entries(
            [_spec(vendor="anthropic", model="claude-opus-5-5", context_window=7)],
            version="test",
        )
        assert context_window("claude", "claude-opus-5-5", table) == 7
        assert context_window("claude", "claude-opus-5-5") == 1_000_000


class TestValidation:
    @pytest.mark.parametrize(
        "overrides",
        [
            {"vendor": ""},
            {"model": " "},
            {"source": "http://example.com"},
            {"context_window": 0},
            {"context_window": True},
            {"context_window": 1.5},
        ],
    )
    def test_a_bad_entry_is_rejected(self, overrides: dict[str, object]) -> None:
        with pytest.raises(ValueError, match=r"\S"):
            _spec(**overrides)

    def test_a_datetime_check_date_is_rejected(self) -> None:
        with pytest.raises(TypeError, match="checked"):
            _spec(checked=dt.datetime(2026, 10, 16, tzinfo=dt.timezone.utc))

    def test_duplicates_and_late_checks_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicate"):
            ModelTable("v", CHECKED, (_spec(), _spec()))
        with pytest.raises(ValueError, match="after the table"):
            ModelTable("v", CHECKED, (_spec(checked=CHECKED + dt.timedelta(days=1)),))


class TestProfileModels:
    @pytest.mark.parametrize("provider", provider_names())
    def test_models_are_unique_and_the_default_is_the_first(self, provider: str) -> None:
        profile = get_provider(provider).profile
        assert len(set(profile.models)) == len(profile.models)
        assert profile.default_model == (profile.models[0] if profile.models else None)

    @pytest.mark.parametrize("provider", provider_names())
    def test_every_listed_model_of_a_known_vendor_has_a_window(self, provider: str) -> None:
        """A model a provider offers is never left without its facts by accident."""
        profile = get_provider(provider).profile
        for model in profile.models:
            if "/" in model and not model.startswith(("anthropic/", "openai/")):
                continue
            assert context_window(provider, model) is not None, (provider, model)

    def test_the_current_claude_models_are_offered(self) -> None:
        models = get_provider("claude").profile.models
        for model in (
            "claude-fable-5-1",
            "claude-opus-5-5",
            "claude-sonnet-5-5",
            "claude-haiku-5-5",
        ):
            assert model in models
