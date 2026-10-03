"""``ConfigScope.PROJECT``: a session loads none of the user's own CLI configuration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from agentshim import (
    CliAgent,
    ConfigScope,
    ProviderCapabilityError,
    get_provider,
    prepare_config_home,
    provider_names,
)
from agentshim.testing import FakeExecutor, scripted_turn
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

_PATH = "/usr/bin:/bin"

#: What a user keeps in a Codex home besides the login: none of it may reach
#: an isolated home.
_OPERATOR_FILES = st.sets(
    st.sampled_from(
        ["AGENTS.md", "AGENTS.override.md", "hooks.json", "config.toml", "memories/MEMORY.md"]
    )
)


def _turn_argv(provider: str, env: dict[str, str]) -> list[str]:
    executor = FakeExecutor(scripted_turn(provider, text="ok"))
    agent = CliAgent(provider, executor=executor, env=env)
    agent.start_session(config_scope=ConfigScope.PROJECT).turn("go")
    return list(executor.requests[0].argv)


def _isolated_env(provider: str, tmp_path: Path) -> dict[str, str]:
    env = {"PATH": _PATH, "HOME": str(tmp_path / "user")}
    profile = get_provider(provider).profile
    return {**env, **prepare_config_home(profile, tmp_path / "run-home", env)}


class TestPolicy:
    @pytest.mark.parametrize("provider", provider_names())
    def test_a_provider_isolates_or_refuses_never_silently_loads(
        self, provider: str, tmp_path: Path
    ) -> None:
        profile = get_provider(provider).profile
        executor = FakeExecutor([scripted_turn(provider, text="ok")])
        agent = CliAgent(provider, executor=executor, env=_isolated_env(provider, tmp_path))
        if ConfigScope.PROJECT not in profile.config_scopes:
            with pytest.raises(ProviderCapabilityError, match="scope 'project'"):
                agent.start_session(config_scope=ConfigScope.PROJECT)
            assert executor.requests == []
            return
        agent.start_session(config_scope=ConfigScope.PROJECT).turn("go")
        default = FakeExecutor([scripted_turn(provider, text="ok")])
        CliAgent(provider, executor=default, env=_isolated_env(provider, tmp_path)).run("go")
        assert list(executor.requests[0].argv) != list(default.requests[0].argv)

    def test_claude_and_codex_both_support_it(self) -> None:
        for provider in ("claude", "codex"):
            assert ConfigScope.PROJECT in get_provider(provider).profile.config_scopes

    def test_the_default_is_all(self) -> None:
        executor = FakeExecutor(scripted_turn("claude", text="ok"))
        session = CliAgent("claude", executor=executor, env={"PATH": _PATH}).start_session()
        assert session.config_scope is ConfigScope.ALL


class TestClaude:
    def test_user_settings_and_auto_memory_are_off(self, tmp_path: Path) -> None:
        argv = _turn_argv("claude", _isolated_env("claude", tmp_path))
        assert argv[argv.index("--setting-sources") + 1] == "project,local"
        assert json.loads(argv[argv.index("--settings") + 1])["autoMemoryEnabled"] is False

    def test_no_home_is_needed(self, tmp_path: Path) -> None:
        env = {"PATH": _PATH, "HOME": str(tmp_path)}
        assert prepare_config_home(get_provider("claude").profile, tmp_path / "h", env) == {}
        assert not (tmp_path / "h").exists()


class TestCodex:
    def test_a_turn_in_the_users_own_home_is_refused(self, tmp_path: Path) -> None:
        env = {"PATH": _PATH, "HOME": str(tmp_path)}
        with pytest.raises(ProviderCapabilityError, match="CODEX_HOME"):
            _turn_argv("codex", env)
        with pytest.raises(ProviderCapabilityError, match="user's own"):
            _turn_argv("codex", {**env, "CODEX_HOME": str(tmp_path / ".codex")})

    def test_a_prepared_home_runs_without_user_config_or_memories(self, tmp_path: Path) -> None:
        argv = _turn_argv("codex", _isolated_env("codex", tmp_path))
        assert "--ignore-user-config" in argv
        assert argv[argv.index("--disable") + 1] == "memories"

    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=25)
    @given(operator_files=_OPERATOR_FILES, refresh=st.booleans())
    def test_the_home_holds_the_login_and_nothing_else_of_the_users(
        self,
        tmp_path_factory: pytest.TempPathFactory,
        *,
        operator_files: set[str],
        refresh: bool,
    ) -> None:
        root = tmp_path_factory.mktemp("case")
        user_home = root / "user" / ".codex"
        user_home.mkdir(parents=True)
        (user_home / "auth.json").write_text('{"token": "a"}')
        for name in operator_files:
            (user_home / name).parent.mkdir(parents=True, exist_ok=True)
            (user_home / name).write_text("OPERATOR")
        env = {"HOME": str(root / "user")}
        profile = get_provider("codex").profile
        home = root / "run-home"
        if refresh:
            prepare_config_home(profile, home, env)
            (user_home / "auth.json").write_text('{"token": "b"}')
        overrides = prepare_config_home(profile, home, env)
        assert overrides == {"CODEX_HOME": str(home)}
        assert sorted(p.name for p in home.rglob("*")) == ["auth.json"]
        assert (home / "auth.json").read_text() == (user_home / "auth.json").read_text()

    def test_the_users_own_root_cannot_be_the_home(self, tmp_path: Path) -> None:
        env = {"HOME": str(tmp_path)}
        with pytest.raises(ValueError, match="user's own state root"):
            prepare_config_home(get_provider("codex").profile, tmp_path / ".codex", env)
        with pytest.raises(ValueError, match="absolute"):
            prepare_config_home(get_provider("codex").profile, Path("rel"), env)
