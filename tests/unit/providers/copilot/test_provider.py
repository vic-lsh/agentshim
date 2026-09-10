"""Copilot profile, MCP installation, and exit classification."""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

from agentshim import (
    CliExitError,
    HttpMcpServer,
    McpMechanism,
    OutputSchemaStyle,
    ProviderProfile,
    StdioMcpServer,
    get_provider,
    provider_names,
)
from agentshim.providers.copilot import CopilotProvider

# Populated by the profile, but empty on purpose: Copilot has no output
# schema, keeps no macOS-only state, and its ``--additional-mcp-config`` flag
# never writes a workspace config file.
EMPTY_BY_DESIGN = {"schema_dialect", "darwin_state_dirs", "mcp_config_file"}


def _stdio() -> StdioMcpServer:
    return StdioMcpServer(
        name="vibesys-issues", command="python", args=["-m", "board.mcp", "issues.json"]
    )


def _config(installation_argv: list[str]) -> dict[str, dict[str, dict[str, object]]]:
    assert installation_argv[0] == "--additional-mcp-config"
    return json.loads(installation_argv[1])


class TestProfile:
    def test_get_provider_returns_a_copilot_provider(self) -> None:
        assert isinstance(get_provider("copilot"), CopilotProvider)

    def test_copilot_is_a_registered_provider_name(self) -> None:
        assert "copilot" in provider_names()

    def test_every_profile_field_is_populated(self) -> None:
        profile = CopilotProvider().profile
        for field in fields(ProviderProfile):
            if field.name in EMPTY_BY_DESIGN:
                continue
            value = getattr(profile, field.name)
            assert value is not None, field.name
            assert value != (), field.name
            assert value != "", field.name

    def test_declared_capabilities(self) -> None:
        profile = CopilotProvider().profile
        assert profile.name == "copilot"
        assert profile.display_name == "Copilot CLI"
        assert profile.binary == "copilot"
        assert profile.supports_resume is True
        assert profile.supports_reasoning_effort is False
        assert profile.mcp is McpMechanism.CLI_FLAGS
        assert profile.output_schema is OutputSchemaStyle.NONE
        assert profile.schema_dialect is None

    def test_state_and_auth_declarations(self) -> None:
        profile = CopilotProvider().profile
        assert ".copilot" in profile.state_dirs
        assert ".config/github-copilot" in profile.state_dirs
        assert all(not path.startswith("/") for path in profile.state_dirs)
        assert profile.darwin_state_dirs == ()
        assert set(profile.auth_env_vars) == {"COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"}
        assert profile.skill_dirs == (".github/skills",)
        assert any("@github/copilot" in command for command in profile.container_install)
        assert profile.state_root_env == "COPILOT_HOME"
        assert profile.auth_files == (".copilot/config.json", ".copilot/settings.json")
        assert profile.mcp_config_file is None
        assert profile.container_env == {}


class TestInstallMcp:
    def test_no_servers_installs_nothing(self) -> None:
        installation = CopilotProvider().install_mcp(None, [])
        assert list(installation.argv) == []
        installation.restore()

    def test_servers_become_one_config_flag(self) -> None:
        installation = CopilotProvider().install_mcp(None, [_stdio()])
        argv = list(installation.argv)
        assert argv[0] == "--additional-mcp-config"
        assert _config(argv)["mcpServers"]["vibesys-issues"] == {
            "command": "python",
            "args": ["-m", "board.mcp", "issues.json"],
        }

    def test_nothing_is_written_to_the_workspace(self, tmp_path: Path) -> None:
        installation = CopilotProvider().install_mcp(tmp_path, [_stdio()])
        assert list(tmp_path.iterdir()) == []
        installation.restore()
        assert list(tmp_path.iterdir()) == []

    def test_a_workspace_is_not_required(self) -> None:
        """The flag carries the servers, so a turn without a cwd still works."""
        assert list(CopilotProvider().install_mcp(None, [_stdio()]).argv)

    def test_env_is_included_only_when_present(self) -> None:
        plain = _config(
            list(CopilotProvider().install_mcp(None, [StdioMcpServer("tool", "npx")]).argv)
        )
        assert "env" not in plain["mcpServers"]["tool"]
        server = StdioMcpServer(name="tool", command="npx", env={"KEY": "val"})
        with_env = _config(list(CopilotProvider().install_mcp(None, [server]).argv))
        assert with_env["mcpServers"]["tool"]["env"] == {"KEY": "val"}

    def test_http_servers_default_to_streamable_http(self) -> None:
        server = HttpMcpServer(name="my-srv", url="http://localhost:9000/mcp")
        config = _config(list(CopilotProvider().install_mcp(None, [server]).argv))
        assert config["mcpServers"]["my-srv"] == {
            "type": "http",
            "url": "http://localhost:9000/mcp",
        }

    def test_an_sse_server_is_declared_as_sse(self) -> None:
        server = HttpMcpServer(name="my-srv", url="http://localhost:9000/sse", transport="sse")
        config = _config(list(CopilotProvider().install_mcp(None, [server]).argv))
        assert config["mcpServers"]["my-srv"] == {"type": "sse", "url": "http://localhost:9000/sse"}

    def test_http_headers_are_included_when_set(self) -> None:
        server = HttpMcpServer(
            name="auth", url="https://x/sse", headers={"Authorization": "Bearer t"}
        )
        config = _config(list(CopilotProvider().install_mcp(None, [server]).argv))
        assert config["mcpServers"]["auth"]["headers"] == {"Authorization": "Bearer t"}

    def test_multiple_servers_are_merged_into_one_flag(self) -> None:
        servers = [_stdio(), HttpMcpServer(name="http-srv", url="http://localhost:8080")]
        argv = list(CopilotProvider().install_mcp(None, servers).argv)
        assert len(argv) == 2
        assert set(_config(argv)["mcpServers"]) == {"vibesys-issues", "http-srv"}

    def test_the_flag_is_valid_json_on_one_line(self) -> None:
        argv = list(CopilotProvider().install_mcp(None, [_stdio()]).argv)
        assert "\n" not in argv[1]
        assert json.loads(argv[1])


class TestClassifyExit:
    def test_a_fresh_turn_keeps_the_exit_error(self) -> None:
        error = CliExitError(["copilot"], 1, "", "boom")
        assert CopilotProvider().classify_exit(error, resumed=False) is error

    def test_a_resumed_turn_keeps_the_exit_error_too(self) -> None:
        """Copilot reports a lost session the way it reports any other failure."""
        error = CliExitError(["copilot", "--resume", "abc-123"], 1, "", "boom")
        assert CopilotProvider().classify_exit(error, resumed=True) is error
