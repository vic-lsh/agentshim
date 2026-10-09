"""What the app-server profile declares, next to the one-shot provider's."""

from __future__ import annotations

from agentshim import ConfigScope, NativeMode, OutputSchemaStyle, get_provider
from agentshim.providers.codex.app_server import APP_SERVER_PROFILE


def test_the_app_server_enforces_every_native_mode_and_the_one_shot_provider_still_only_bypass() -> (
    None
):
    assert APP_SERVER_PROFILE.native_permission_modes == frozenset(NativeMode)
    assert get_provider("codex").profile.native_permission_modes == frozenset({NativeMode.BYPASS})


def test_the_renewal_budget_is_the_one_shot_providers() -> None:
    assert APP_SERVER_PROFILE.renewal == get_provider("codex").profile.renewal
    assert APP_SERVER_PROFILE.renewal is not None
    assert APP_SERVER_PROFILE.renewal.max_turns == 2


def test_the_schema_travels_inline_and_a_dedicated_home_is_not_offered() -> None:
    assert APP_SERVER_PROFILE.output_schema is OutputSchemaStyle.INLINE_JSON
    assert APP_SERVER_PROFILE.config_scopes == frozenset({ConfigScope.ALL})
    assert get_provider("codex").profile.output_schema is OutputSchemaStyle.FILE_PATH


def test_it_is_still_codex() -> None:
    assert APP_SERVER_PROFILE.name == get_provider("codex").profile.name
    assert APP_SERVER_PROFILE.binary == "codex"
