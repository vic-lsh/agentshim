"""opencode profile, MCP installation, and exit classification."""

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
from agentshim.providers.opencode import OpencodeProvider


def _stdio() -> StdioMcpServer:
    return StdioMcpServer(
        name="vibesys-issues", command="python", args=["-m", "board.mcp", "issues.json"]
    )


def _config(workspace: Path) -> dict[str, object]:
    return json.loads((workspace / "opencode.json").read_text())


class TestProfile:
    def test_get_provider_returns_an_opencode_provider(self) -> None:
        assert isinstance(get_provider("opencode"), OpencodeProvider)

    def test_every_profile_field_is_populated(self) -> None:
        profile = OpencodeProvider().profile
        optional = {"schema_dialect", "darwin_state_dirs", "auth_env_vars"}
        for field in fields(ProviderProfile):
            value = getattr(profile, field.name)
            if field.name in optional:
                continue
            assert value is not None, field.name
            assert value != (), field.name
            assert value != "", field.name

    def test_declared_capabilities(self) -> None:
        profile = OpencodeProvider().profile
        assert profile.name == "opencode"
        assert profile.display_name == "opencode"
        assert profile.binary == "opencode"
        assert profile.supports_resume is True
        assert profile.supports_reasoning_effort is False
        assert profile.mcp is McpMechanism.CONFIG_FILE
        assert profile.output_schema is OutputSchemaStyle.NONE
        assert profile.schema_dialect is None

    def test_state_and_auth_declarations(self) -> None:
        profile = OpencodeProvider().profile
        assert profile.state_dirs == (".local/share/opencode", ".config/opencode")
        assert all(not path.startswith("/") for path in profile.state_dirs)
        assert profile.darwin_state_dirs == ()
        assert profile.auth_env_vars == ()
        assert profile.skill_dirs == (".opencode/skills",)

    def test_the_container_install_runs_the_official_installer(self) -> None:
        commands = OpencodeProvider().profile.container_install
        assert any("https://opencode.ai/install" in command for command in commands)
        assert any("/usr/local/bin/opencode" in command for command in commands)


class TestInstallMcp:
    def test_no_servers_installs_nothing(self, tmp_path: Path) -> None:
        installation = OpencodeProvider().install_mcp(tmp_path, [])
        assert list(installation.argv) == []
        installation.restore()
        assert list(tmp_path.iterdir()) == []

    def test_servers_land_in_opencode_json(self, tmp_path: Path) -> None:
        installation = OpencodeProvider().install_mcp(tmp_path, [_stdio()])
        assert _config(tmp_path)["mcp"]["vibesys-issues"] == {  # pyright: ignore[reportIndexIssue]
            "type": "local",
            "command": ["python", "-m", "board.mcp", "issues.json"],
            "enabled": True,
        }
        assert list(installation.argv) == []

    def test_the_schema_default_is_written(self, tmp_path: Path) -> None:
        OpencodeProvider().install_mcp(tmp_path, [_stdio()])
        assert _config(tmp_path)["$schema"] == "https://opencode.ai/config.json"

    def test_env_is_included_only_when_present(self, tmp_path: Path) -> None:
        OpencodeProvider().install_mcp(
            tmp_path, [StdioMcpServer(name="tool", command="npx", env={"KEY": "val"})]
        )
        assert _config(tmp_path)["mcp"]["tool"]["environment"] == {"KEY": "val"}  # pyright: ignore[reportIndexIssue]

    def test_http_servers_declare_the_remote_transport(self, tmp_path: Path) -> None:
        OpencodeProvider().install_mcp(
            tmp_path, [HttpMcpServer(name="my-srv", url="http://localhost:9000/mcp")]
        )
        assert _config(tmp_path)["mcp"]["my-srv"] == {  # pyright: ignore[reportIndexIssue]
            "type": "remote",
            "url": "http://localhost:9000/mcp",
            "enabled": True,
        }

    def test_both_transports_render_the_same_remote_entry(self, tmp_path: Path) -> None:
        """opencode has one remote type and negotiates the transport itself."""
        for transport in ("http", "sse"):
            OpencodeProvider().install_mcp(
                tmp_path,
                [HttpMcpServer(name="my-srv", url="http://localhost:9000/x", transport=transport)],  # pyright: ignore[reportArgumentType]
            )
            assert _config(tmp_path)["mcp"]["my-srv"] == {  # pyright: ignore[reportIndexIssue]
                "type": "remote",
                "url": "http://localhost:9000/x",
                "enabled": True,
            }

    def test_http_headers_are_included_when_set(self, tmp_path: Path) -> None:
        server = HttpMcpServer(
            name="auth", url="https://x/mcp", headers={"Authorization": "Bearer t"}
        )
        OpencodeProvider().install_mcp(tmp_path, [server])
        assert _config(tmp_path)["mcp"]["auth"]["headers"] == {"Authorization": "Bearer t"}  # pyright: ignore[reportIndexIssue]

    def test_multiple_servers_are_merged(self, tmp_path: Path) -> None:
        servers = [_stdio(), HttpMcpServer(name="http-srv", url="http://localhost:8080")]
        OpencodeProvider().install_mcp(tmp_path, servers)
        assert set(_config(tmp_path)["mcp"]) == {"vibesys-issues", "http-srv"}  # pyright: ignore[reportArgumentType]

    def test_restore_removes_a_config_it_created(self, tmp_path: Path) -> None:
        installation = OpencodeProvider().install_mcp(tmp_path, [_stdio()])
        installation.restore()
        assert not (tmp_path / "opencode.json").exists()

    def test_restore_puts_back_the_users_own_config(self, tmp_path: Path) -> None:
        target = tmp_path / "opencode.json"
        original = json.dumps(
            {"$schema": "https://opencode.ai/config.json", "model": "x/y"}, indent=2
        )
        target.write_text(original)

        installation = OpencodeProvider().install_mcp(tmp_path, [_stdio()])
        assert "vibesys-issues" in _config(tmp_path)["mcp"]  # pyright: ignore[reportOperatorIssue]
        installation.restore()
        assert target.read_text() == original

    def test_a_workspace_is_required(self) -> None:
        with pytest.raises(ProviderCapabilityError, match="cwd"):
            OpencodeProvider().install_mcp(None, [_stdio()])


class TestClassifyExit:
    def test_a_fresh_turn_keeps_the_exit_error(self) -> None:
        error = CliExitError(["opencode", "run"], 1, "", "boom")
        assert OpencodeProvider().classify_exit(error, resumed=False) is error

    def test_a_resumed_turn_keeps_the_exit_error_too(self) -> None:
        """opencode gives no distinguishable exit code for a refused resume."""
        error = CliExitError(["opencode", "run", "--session", "ses_1"], 1, "", "session not found")
        assert OpencodeProvider().classify_exit(error, resumed=True) is error
