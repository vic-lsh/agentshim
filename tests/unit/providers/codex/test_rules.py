"""Codex exec-policy rules rendered from ``excluded_commands``.

The rules file is a contract with Codex's Starlark loader, so the properties
check it against ``parse_rules``, its inverse; the e2e suite checks the same
rendering against ``codex execpolicy check``.
"""

from __future__ import annotations

import shlex
from typing import TYPE_CHECKING

import pytest
from agentshim.providers.codex import (
    RULES_FILENAME,
    CodexSandboxConfig,
    install_rules,
    parse_rules,
    render_rules,
)
from hypothesis import HealthCheck, given, settings

from tests.unit.providers.codex.test_sandbox import exempting

if TYPE_CHECKING:
    from pathlib import Path


class TestRender:
    @given(exempting)
    def test_parse_rules_recovers_every_command_word_for_word(
        self, config: CodexSandboxConfig
    ) -> None:
        """No word can close its string literal and smuggle in another rule."""
        expected = [shlex.split(command) for command in config.excluded_commands]
        assert parse_rules(render_rules(config)) == expected

    @given(exempting)
    def test_every_rule_is_an_allow_prefix_rule(self, config: CodexSandboxConfig) -> None:
        lines = [line for line in render_rules(config).splitlines() if not line.startswith("#")]
        assert len(lines) == len(config.excluded_commands)
        for line in lines:
            assert line.startswith("prefix_rule(pattern=[")
            assert line.endswith('], decision="allow")')

    def test_the_gateway_command_renders_as_one_readable_rule(self) -> None:
        config = CodexSandboxConfig(excluded_commands=["sdo detector check"])
        rules = [line for line in render_rules(config).splitlines() if line[:1] != "#"]
        assert rules == ['prefix_rule(pattern=["sdo", "detector", "check"], decision="allow")']

    def test_no_exemptions_render_no_rules(self) -> None:
        assert parse_rules(render_rules(CodexSandboxConfig())) == []

    @pytest.mark.parametrize(
        "text", ['prefix_rule(pattern=["x"], decision="prompt")', "host_executable(name='x')"]
    )
    def test_parse_rules_rejects_what_render_rules_never_writes(self, text: str) -> None:
        with pytest.raises(ValueError, match="not a rule agentshim writes"):
            parse_rules(text)


class TestInstall:
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(exempting)
    def test_the_file_holds_exactly_the_rendered_rules(
        self, tmp_path: Path, config: CodexSandboxConfig
    ) -> None:
        path = install_rules(tmp_path / "home", config)
        assert path == tmp_path / "home" / "rules" / RULES_FILENAME
        assert path.read_text(encoding="utf-8") == render_rules(config)
        assert [p.name for p in path.parent.iterdir()] == [RULES_FILENAME]

    def test_reinstalling_replaces_the_previous_rules(self, tmp_path: Path) -> None:
        install_rules(tmp_path, CodexSandboxConfig(excluded_commands=["old"]))
        path = install_rules(tmp_path, CodexSandboxConfig(excluded_commands=["new"]))
        assert parse_rules(path.read_text(encoding="utf-8")) == [["new"]]

    def test_another_rules_file_is_refused_and_left_alone(self, tmp_path: Path) -> None:
        """Codex would load it too, exempting commands the config does not name."""
        rules = tmp_path / "rules"
        rules.mkdir()
        (rules / "default.rules").write_text('prefix_rule(pattern=["bash"], decision="allow")\n')
        with pytest.raises(ValueError, match=r"default\.rules.*dedicated CODEX_HOME"):
            install_rules(tmp_path, CodexSandboxConfig(excluded_commands=["sdo check"]))
        assert sorted(p.name for p in rules.iterdir()) == ["default.rules"]

    def test_files_codex_does_not_load_are_tolerated(self, tmp_path: Path) -> None:
        (tmp_path / "rules").mkdir()
        (tmp_path / "rules" / "README").write_text("notes")
        install_rules(tmp_path, CodexSandboxConfig(excluded_commands=["sdo check"]))
        assert (tmp_path / "rules" / "README").exists()
