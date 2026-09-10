"""Claude profile, MCP installation, and exit classification."""

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
    SchemaDialect,
    SessionResumeError,
    StdioMcpServer,
    get_provider,
)
from agentshim.providers.claude import ClaudeProvider


def _stdio() -> StdioMcpServer:
    return StdioMcpServer(
        name="vibesys-issues", command="python", args=["-m", "board.mcp", "issues.json"]
    )


class TestProfile:
    def test_get_provider_returns_a_claude_provider(self) -> None:
        assert isinstance(get_provider("claude"), ClaudeProvider)

    def test_unknown_provider_names_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown provider"):
            get_provider("nope")

    def test_every_profile_field_is_populated(self) -> None:
        profile = ClaudeProvider().profile
        for field in fields(ProviderProfile):
            value = getattr(profile, field.name)
            assert value is not None, field.name
            if field.name != "schema_dialect":
                assert value != (), field.name
                assert value != "", field.name

    def test_declared_capabilities(self) -> None:
        profile = ClaudeProvider().profile
        assert profile.name == "claude"
        assert profile.display_name == "Claude Code"
        assert profile.binary == "claude"
        assert profile.supports_resume is True
        assert profile.supports_reasoning_effort is True
        assert profile.mcp is McpMechanism.CONFIG_FILE
        assert profile.output_schema is OutputSchemaStyle.INLINE_JSON
        assert profile.schema_dialect is SchemaDialect.OPEN

    def test_state_and_auth_declarations(self) -> None:
        profile = ClaudeProvider().profile
        assert ".claude" in profile.state_dirs
        assert ".claude.json" in profile.state_dirs
        assert all(not path.startswith("/") for path in profile.state_dirs)
        assert all(path.startswith("Library/") for path in profile.darwin_state_dirs)
        assert "ANTHROPIC_API_KEY" in profile.auth_env_vars
        assert profile.skill_dirs == (".claude/skills",)
        assert any("claude.ai/install.sh" in command for command in profile.container_install)


class TestInstallMcp:
    def test_no_servers_installs_nothing(self, tmp_path: Path) -> None:
        installation = ClaudeProvider().install_mcp(tmp_path, [])
        assert list(installation.argv) == []
        installation.restore()
        assert list(tmp_path.iterdir()) == []

    def test_servers_land_in_dot_mcp_json(self, tmp_path: Path) -> None:
        installation = ClaudeProvider().install_mcp(tmp_path, [_stdio()])
        config = json.loads((tmp_path / ".mcp.json").read_text())
        server = config["mcpServers"]["vibesys-issues"]
        assert server == {"command": "python", "args": ["-m", "board.mcp", "issues.json"]}
        assert list(installation.argv) == []
        installation.restore()
        assert not (tmp_path / ".mcp.json").exists()

    def test_env_is_included_only_when_present(self, tmp_path: Path) -> None:
        server = StdioMcpServer(name="tool", command="npx", env={"KEY": "val"})
        ClaudeProvider().install_mcp(tmp_path, [server])
        config = json.loads((tmp_path / ".mcp.json").read_text())
        assert config["mcpServers"]["tool"]["env"] == {"KEY": "val"}

    def test_http_servers_default_to_streamable_http(self, tmp_path: Path) -> None:
        """Claude validates the config strictly and needs an explicit type."""
        server = HttpMcpServer(name="my-srv", url="http://localhost:9000/mcp")
        ClaudeProvider().install_mcp(tmp_path, [server])
        config = json.loads((tmp_path / ".mcp.json").read_text())
        assert config["mcpServers"]["my-srv"] == {
            "type": "http",
            "url": "http://localhost:9000/mcp",
        }

    def test_an_sse_server_is_declared_as_sse(self, tmp_path: Path) -> None:
        server = HttpMcpServer(name="my-srv", url="http://localhost:9000/sse", transport="sse")
        ClaudeProvider().install_mcp(tmp_path, [server])
        config = json.loads((tmp_path / ".mcp.json").read_text())
        assert config["mcpServers"]["my-srv"] == {"type": "sse", "url": "http://localhost:9000/sse"}

    def test_http_headers_are_included_when_set(self, tmp_path: Path) -> None:
        server = HttpMcpServer(
            name="auth", url="https://x/sse", headers={"Authorization": "Bearer t"}
        )
        ClaudeProvider().install_mcp(tmp_path, [server])
        config = json.loads((tmp_path / ".mcp.json").read_text())
        assert config["mcpServers"]["auth"]["headers"] == {"Authorization": "Bearer t"}

    def test_multiple_servers_are_merged(self, tmp_path: Path) -> None:
        servers = [_stdio(), HttpMcpServer(name="http-srv", url="http://localhost:8080")]
        ClaudeProvider().install_mcp(tmp_path, servers)
        config = json.loads((tmp_path / ".mcp.json").read_text())
        assert set(config["mcpServers"]) == {"vibesys-issues", "http-srv"}

    def test_a_workspace_is_required(self) -> None:
        with pytest.raises(ProviderCapabilityError, match="cwd"):
            ClaudeProvider().install_mcp(None, [_stdio()])


class TestClassifyExit:
    def test_a_fresh_turn_keeps_the_exit_error(self) -> None:
        error = CliExitError(["claude", "-p"], 1, "", "boom")
        assert ClaudeProvider().classify_exit(error, resumed=False) is error

    def test_any_nonzero_exit_on_a_resumed_turn_means_the_session_is_gone(self) -> None:
        """``claude --resume`` gives no distinguishable code for a lost transcript."""
        error = CliExitError(
            ["claude", "--resume", "abc-123", "-p"], 1, "", "no conversation found"
        )
        classified = ClaudeProvider().classify_exit(error, resumed=True)
        assert isinstance(classified, SessionResumeError)
        assert classified.session_id == "abc-123"
        assert classified.returncode == 1
        assert classified.stderr == "no conversation found"

    def test_a_resumed_turn_without_the_flag_in_argv_stays_generic(self) -> None:
        error = CliExitError(["claude", "-p"], 1, "", "boom")
        assert ClaudeProvider().classify_exit(error, resumed=True) is error

    def test_session_resume_failed_is_an_exit_error(self) -> None:
        error = CliExitError(["claude", "--resume", "s1"], 2)
        classified = ClaudeProvider().classify_exit(error, resumed=True)
        assert isinstance(classified, CliExitError)
        assert "s1" in str(classified)
