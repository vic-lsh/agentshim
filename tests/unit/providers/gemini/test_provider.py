"""Gemini profile, MCP installation, and exit classification."""

from __future__ import annotations

import json
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
    StdioMcpServer,
    get_provider,
)
from agentshim.providers.gemini import GeminiProvider

SETTINGS = ("gemini", "settings.json")


def _stdio() -> StdioMcpServer:
    return StdioMcpServer(
        name="vibesys-issues", command="python", args=["-m", "board.mcp", "issues.json"]
    )


def _settings(workspace: Path) -> dict[str, object]:
    return json.loads((workspace / ".gemini" / "settings.json").read_text())


class TestProfile:
    def test_get_provider_returns_a_gemini_provider(self) -> None:
        assert isinstance(get_provider("gemini"), GeminiProvider)

    def test_every_profile_field_is_populated(self) -> None:
        profile = GeminiProvider().profile
        optional = {"schema_dialect", "darwin_state_dirs"}
        for field in fields(ProviderProfile):
            value = getattr(profile, field.name)
            if field.name in optional:
                continue
            assert value is not None, field.name
            assert value != (), field.name
            assert value != "", field.name

    def test_declared_capabilities(self) -> None:
        profile = GeminiProvider().profile
        assert profile.name == "gemini"
        assert profile.display_name == "Gemini CLI"
        assert profile.binary == "gemini"
        assert profile.supports_resume is True
        assert profile.supports_reasoning_effort is False
        assert profile.mcp is McpMechanism.CONFIG_FILE
        assert profile.output_schema is OutputSchemaStyle.NONE
        assert profile.schema_dialect is None

    def test_state_and_auth_declarations(self) -> None:
        profile = GeminiProvider().profile
        assert profile.state_dirs == (".gemini", ".config/gemini")
        assert all(not path.startswith("/") for path in profile.state_dirs)
        assert profile.darwin_state_dirs == ()
        assert profile.auth_env_vars == ("GEMINI_API_KEY", "GOOGLE_API_KEY")
        assert profile.skill_dirs == (".gemini/skills",)

    def test_the_container_install_brings_node_and_the_cli(self) -> None:
        commands = GeminiProvider().profile.container_install
        assert any("nodejs.org" in command for command in commands)
        assert any("npm install -g @google/gemini-cli" in command for command in commands)


class TestInstallMcp:
    def test_no_servers_installs_nothing(self, tmp_path: Path) -> None:
        installation = GeminiProvider().install_mcp(tmp_path, [])
        assert list(installation.argv) == []
        installation.restore()
        assert list(tmp_path.iterdir()) == []

    def test_servers_land_in_the_workspace_settings(self, tmp_path: Path) -> None:
        installation = GeminiProvider().install_mcp(tmp_path, [_stdio()])
        server = _settings(tmp_path)["mcpServers"]["vibesys-issues"]  # pyright: ignore[reportIndexIssue]
        assert server == {
            "command": "python",
            "args": ["-m", "board.mcp", "issues.json"],
            "trust": True,
        }
        assert list(installation.argv) == []

    def test_trust_skips_the_per_tool_approval_prompt(self, tmp_path: Path) -> None:
        GeminiProvider().install_mcp(tmp_path, [_stdio()])
        assert _settings(tmp_path)["mcpServers"]["vibesys-issues"]["trust"] is True  # pyright: ignore[reportIndexIssue]

    def test_env_is_included_only_when_present(self, tmp_path: Path) -> None:
        GeminiProvider().install_mcp(
            tmp_path, [StdioMcpServer(name="tool", command="npx", env={"KEY": "val"})]
        )
        assert _settings(tmp_path)["mcpServers"]["tool"]["env"] == {"KEY": "val"}  # pyright: ignore[reportIndexIssue]

    def test_http_servers_use_the_streamable_http_key(self, tmp_path: Path) -> None:
        GeminiProvider().install_mcp(
            tmp_path, [HttpMcpServer(name="my-srv", url="http://localhost:9000/mcp")]
        )
        assert _settings(tmp_path)["mcpServers"]["my-srv"] == {  # pyright: ignore[reportIndexIssue]
            "httpUrl": "http://localhost:9000/mcp",
            "trust": True,
        }

    def test_http_headers_are_included_when_set(self, tmp_path: Path) -> None:
        server = HttpMcpServer(
            name="auth", url="https://x/mcp", headers={"Authorization": "Bearer t"}
        )
        GeminiProvider().install_mcp(tmp_path, [server])
        assert _settings(tmp_path)["mcpServers"]["auth"]["headers"] == {"Authorization": "Bearer t"}  # pyright: ignore[reportIndexIssue]

    def test_multiple_servers_are_merged(self, tmp_path: Path) -> None:
        servers = [_stdio(), HttpMcpServer(name="http-srv", url="http://localhost:8080")]
        GeminiProvider().install_mcp(tmp_path, servers)
        assert set(_settings(tmp_path)["mcpServers"]) == {"vibesys-issues", "http-srv"}  # pyright: ignore[reportArgumentType]

    def test_restore_removes_a_settings_file_it_created(self, tmp_path: Path) -> None:
        installation = GeminiProvider().install_mcp(tmp_path, [_stdio()])
        installation.restore()
        assert not (tmp_path / ".gemini" / "settings.json").exists()

    def test_restore_puts_back_the_users_own_settings(self, tmp_path: Path) -> None:
        target = tmp_path.joinpath(*SETTINGS)
        target.parent.mkdir()
        original = json.dumps({"general": {"previewFeatures": True}}, indent=2)
        target.write_text(original)

        installation = GeminiProvider().install_mcp(tmp_path, [_stdio()])
        assert "vibesys-issues" in _settings(tmp_path)["mcpServers"]  # pyright: ignore[reportOperatorIssue]
        installation.restore()
        assert target.read_text() == original

    def test_a_workspace_is_required(self) -> None:
        with pytest.raises(ProviderCapabilityError, match="cwd"):
            GeminiProvider().install_mcp(None, [_stdio()])


class TestClassifyExit:
    def test_a_fresh_turn_keeps_the_exit_error(self) -> None:
        error = CliExitError(["gemini", "--yolo"], 1, "", "boom")
        assert GeminiProvider().classify_exit(error, resumed=False) is error

    def test_a_resumed_turn_keeps_the_exit_error_too(self) -> None:
        """Gemini gives no distinguishable exit code for a refused resume."""
        error = CliExitError(["gemini", "--resume", "abc-123"], 1, "", "no session found")
        assert GeminiProvider().classify_exit(error, resumed=True) is error
