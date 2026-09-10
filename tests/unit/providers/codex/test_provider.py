"""Codex profile, MCP installation, and exit classification."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pytest
from agentshim import (
    CliExitError,
    HttpMcpServer,
    McpMechanism,
    OutputSchemaStyle,
    ProviderCapabilityError,
    ProviderProfile,
    SchemaDialect,
    SessionResumeError,
    StdioMcpServer,
    get_provider,
)
from agentshim.providers.codex import CodexProvider

_RESUME_ARGV = ["codex", "exec", "resume", "thread-123", "-", "--json"]
_ROLLOUT_GONE = "thread/resume failed: no rollout found for thread id thread-123"


def _stdio() -> StdioMcpServer:
    return StdioMcpServer(
        name="vibesys-issues", command="python", args=["-m", "board.mcp", "issues.json"]
    )


def _flags(*servers: StdioMcpServer | HttpMcpServer) -> list[str]:
    return list(CodexProvider().install_mcp(None, list(servers)).argv)


class TestProfile:
    def test_get_provider_returns_a_codex_provider(self) -> None:
        assert isinstance(get_provider("codex"), CodexProvider)

    def test_every_profile_field_is_populated(self) -> None:
        profile = CodexProvider().profile
        for field in fields(ProviderProfile):
            value = getattr(profile, field.name)
            assert value is not None, field.name
            if field.name != "schema_dialect":
                assert value != (), field.name
                assert value != "", field.name

    def test_declared_capabilities(self) -> None:
        profile = CodexProvider().profile
        assert profile.name == "codex"
        assert profile.display_name == "Codex"
        assert profile.binary == "codex"
        assert profile.supports_resume is True
        assert profile.supports_reasoning_effort is True
        assert profile.mcp is McpMechanism.CLI_FLAGS
        assert profile.output_schema is OutputSchemaStyle.FILE_PATH
        assert profile.schema_dialect is SchemaDialect.STRICT

    def test_state_and_auth_declarations(self) -> None:
        profile = CodexProvider().profile
        assert profile.state_dirs == (".codex", ".config/codex")
        assert all(not path.startswith("/") for path in profile.state_dirs)
        assert "Library/Application Support/com.openai.codex" in profile.darwin_state_dirs
        assert all(path.startswith("Library/") for path in profile.darwin_state_dirs)
        assert profile.auth_env_vars == ("OPENAI_API_KEY", "OPENAI_BASE_URL")
        assert profile.skill_dirs == (".agents/skills",)

    def test_the_container_recipe_installs_node_then_the_pinned_cli(self) -> None:
        commands = CodexProvider().profile.container_install
        assert "nodejs.org" in commands[0]
        assert commands[-1] == "npm install -g --include=optional @openai/codex@0.144.4"


class TestInstallMcp:
    def test_no_servers_installs_nothing(self) -> None:
        assert _flags() == []

    def test_a_stdio_server_becomes_toml_config_overrides(self) -> None:
        assert _flags(_stdio()) == [
            "--config",
            'mcp_servers.vibesys_issues.command="python"',
            "--config",
            'mcp_servers.vibesys_issues.args=["-m","board.mcp","issues.json"]',
        ]

    def test_a_dash_in_the_name_becomes_a_snake_case_table_key(self) -> None:
        """TOML table keys are snake_case by convention."""
        assert all("vibesys-issues" not in flag for flag in _flags(_stdio()))

    def test_env_entries_get_their_own_override(self) -> None:
        server = StdioMcpServer(name="tool", command="npx", env={"KEY": "val"})
        assert _flags(server)[-2:] == ["--config", 'mcp_servers.tool.env.KEY="val"']

    def test_quotes_in_a_value_are_toml_escaped(self) -> None:
        server = StdioMcpServer(name="tool", command='py"thon')
        assert 'mcp_servers.tool.command="py\\"thon"' in _flags(server)

    def test_an_http_server_declares_its_url(self) -> None:
        server = HttpMcpServer(name="my-srv", url="http://localhost:9000/sse")
        assert _flags(server) == ["--config", 'mcp_servers.my_srv.url="http://localhost:9000/sse"']

    def test_http_headers_are_refused_rather_than_dropped(self) -> None:
        server = HttpMcpServer(
            name="auth", url="https://x/sse", headers={"Authorization": "Bearer t"}
        )
        with pytest.raises(ProviderCapabilityError, match="HTTP headers"):
            _flags(server)

    def test_multiple_servers_are_concatenated(self) -> None:
        flags = _flags(_stdio(), HttpMcpServer(name="http-srv", url="http://localhost:8080"))
        assert 'mcp_servers.vibesys_issues.command="python"' in flags
        assert 'mcp_servers.http_srv.url="http://localhost:8080"' in flags

    def test_nothing_is_written_to_the_workspace(self, tmp_path: Path) -> None:
        """Codex has no project-scoped config file; the servers live in argv."""
        installation = CodexProvider().install_mcp(tmp_path, [_stdio()])
        installation.restore()
        assert list(tmp_path.iterdir()) == []


class TestClassifyExit:
    def test_a_fresh_turn_keeps_the_exit_error(self) -> None:
        error = CliExitError(["codex", "exec"], 1, "", _ROLLOUT_GONE)
        assert CodexProvider().classify_exit(error, resumed=False) is error

    def test_a_missing_rollout_means_the_thread_is_gone(self) -> None:
        error = CliExitError(_RESUME_ARGV, 1, "", _ROLLOUT_GONE)
        classified = CodexProvider().classify_exit(error, resumed=True)
        assert isinstance(classified, SessionResumeError)
        assert classified.session_id == "thread-123"
        assert classified.returncode == 1
        assert classified.stderr == _ROLLOUT_GONE

    def test_another_failure_on_a_resumed_turn_stays_generic(self) -> None:
        """Codex names a lost rollout explicitly, so nothing else is guessed."""
        error = CliExitError(_RESUME_ARGV, 1, "", "stream error: 429 too many requests")
        assert CodexProvider().classify_exit(error, resumed=True) is error

    def test_half_the_marker_is_not_enough(self) -> None:
        error = CliExitError(_RESUME_ARGV, 1, "", "thread/resume failed: permission denied")
        assert CodexProvider().classify_exit(error, resumed=True) is error

    def test_a_resumed_turn_without_the_subcommand_in_argv_stays_generic(self) -> None:
        error = CliExitError(["codex", "exec"], 1, "", _ROLLOUT_GONE)
        assert CodexProvider().classify_exit(error, resumed=True) is error

    def test_session_resume_failed_is_an_exit_error(self) -> None:
        error = CliExitError(_RESUME_ARGV, 2, "", _ROLLOUT_GONE)
        classified = CodexProvider().classify_exit(error, resumed=True)
        assert isinstance(classified, CliExitError)
        assert "thread-123" in str(classified)
