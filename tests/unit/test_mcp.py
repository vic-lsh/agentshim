"""MCP server specs and the JSON config-file merge/restore."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentshim import (
    ConfigFileInstallation,
    FlagsInstallation,
    HttpMcpServer,
    McpConfigError,
    NoopInstallation,
    StdioMcpServer,
    install_config_file,
)

_SERVERS = {"vibesys-issues": {"command": "python", "args": ["-m", "board.mcp", "issues.json"]}}


def _install(target: Path) -> ConfigFileInstallation:
    return install_config_file(target, server_key="mcpServers", servers=_SERVERS)


class TestServerSpecs:
    def test_stdio_defaults(self) -> None:
        server = StdioMcpServer(name="tool", command="npx")
        assert server.args == ()
        assert server.env == {}

    def test_http_defaults(self) -> None:
        server = HttpMcpServer(name="h", url="http://localhost:8080/sse")
        assert server.headers == {}

    def test_specs_are_frozen(self) -> None:
        server = StdioMcpServer(name="tool", command="npx")
        with pytest.raises(AttributeError):
            server.name = "other"  # type: ignore[misc]

    @pytest.mark.parametrize(
        "factory",
        [
            lambda: StdioMcpServer(name="", command="npx"),
            lambda: StdioMcpServer(name="tool", command=""),
            lambda: HttpMcpServer(name="", url="http://x"),
            lambda: HttpMcpServer(name="h", url=""),
        ],
    )
    def test_empty_required_fields_are_rejected(self, factory: object) -> None:
        with pytest.raises(ValueError, match="must"):
            factory()  # pyright: ignore[reportCallIssue]


class TestTrivialInstallations:
    def test_noop_has_no_argv_and_restores_idempotently(self) -> None:
        installation = NoopInstallation()
        assert list(installation.argv) == []
        installation.restore()
        installation.restore()

    def test_flags_carry_argv(self) -> None:
        installation = FlagsInstallation(["--config", "mcp_servers.a.command=python"])
        assert list(installation.argv) == ["--config", "mcp_servers.a.command=python"]
        installation.restore()


class TestInstallAndRestore:
    def test_install_creates_the_file(self, tmp_path: Path) -> None:
        target = tmp_path / ".mcp.json"
        _install(target)
        config = json.loads(target.read_text())
        assert config["mcpServers"]["vibesys-issues"]["command"] == "python"

    def test_restore_removes_a_file_we_created(self, tmp_path: Path) -> None:
        target = tmp_path / ".mcp.json"
        _install(target).restore()
        assert not target.exists()

    def test_restore_is_idempotent(self, tmp_path: Path) -> None:
        target = tmp_path / ".mcp.json"
        installation = _install(target)
        installation.restore()
        installation.restore()
        assert not target.exists()

    def test_merge_preserves_and_restores_the_original_bytes(self, tmp_path: Path) -> None:
        target = tmp_path / ".mcp.json"
        original = b'{"permissions":{"allow":["Bash(*)"]},"mcpServers":{"existing":{"command":"user"}}}\n'
        target.write_bytes(original)

        installation = _install(target)
        config = json.loads(target.read_text())
        assert config["permissions"] == {"allow": ["Bash(*)"]}
        assert config["mcpServers"]["existing"] == {"command": "user"}
        assert "vibesys-issues" in config["mcpServers"]

        installation.restore()
        assert target.read_bytes() == original

    def test_defaults_are_only_added_when_missing(self, tmp_path: Path) -> None:
        target = tmp_path / "opencode.json"
        target.write_text('{"$schema":"user-set"}')
        installation = install_config_file(
            target,
            server_key="mcp",
            servers=_SERVERS,
            defaults={"$schema": "https://opencode.ai/config.json"},
        )
        assert json.loads(target.read_text())["$schema"] == "user-set"
        installation.restore()
        assert json.loads(target.read_text()) == {"$schema": "user-set"}

    def test_defaults_are_removed_on_restore(self, tmp_path: Path) -> None:
        target = tmp_path / "opencode.json"
        target.write_text('{"theme":"dark"}')
        installation = install_config_file(
            target,
            server_key="mcp",
            servers=_SERVERS,
            defaults={"$schema": "https://opencode.ai/config.json"},
        )
        assert json.loads(target.read_text())["$schema"] == "https://opencode.ai/config.json"
        installation.restore()
        assert json.loads(target.read_text()) == {"theme": "dark"}


class TestFileEditedDuringTheTurn:
    def test_unrelated_edits_survive_restore(self, tmp_path: Path) -> None:
        target = tmp_path / ".mcp.json"
        target.write_text('{"theme":"light","mcpServers":{"existing":{"command":"user"}}}')
        installation = _install(target)

        config = json.loads(target.read_text())
        config["theme"] = "dark"
        config["new_setting"] = True
        target.write_text(json.dumps(config))

        installation.restore()
        assert json.loads(target.read_text()) == {
            "theme": "dark",
            "new_setting": True,
            "mcpServers": {"existing": {"command": "user"}},
        }

    def test_an_edit_to_an_injected_server_is_kept(self, tmp_path: Path) -> None:
        target = tmp_path / ".mcp.json"
        target.write_text('{"mcpServers":{"vibesys-issues":{"command":"user"}}}')
        installation = _install(target)

        config = json.loads(target.read_text())
        config["mcpServers"]["vibesys-issues"] = {"command": "agent-edited"}
        target.write_text(json.dumps(config))

        installation.restore()
        assert json.loads(target.read_text())["mcpServers"]["vibesys-issues"] == {"command": "agent-edited"}

    def test_a_server_added_during_the_turn_is_kept(self, tmp_path: Path) -> None:
        target = tmp_path / ".mcp.json"
        installation = _install(target)

        config = json.loads(target.read_text())
        config["mcpServers"]["agent-added"] = {"command": "new"}
        target.write_text(json.dumps(config))

        installation.restore()
        assert json.loads(target.read_text()) == {"mcpServers": {"agent-added": {"command": "new"}}}

    def test_deleting_the_file_during_the_turn_is_not_undone(self, tmp_path: Path) -> None:
        target = tmp_path / ".mcp.json"
        target.write_text('{"theme":"dark"}')
        installation = _install(target)
        target.unlink()

        installation.restore()
        assert not target.exists()


class TestFailureModes:
    def test_invalid_json_config_raises(self, tmp_path: Path) -> None:
        target = tmp_path / ".mcp.json"
        target.write_text("{not json")
        with pytest.raises(McpConfigError, match="invalid JSON"):
            _install(target)

    def test_non_object_config_raises(self, tmp_path: Path) -> None:
        target = tmp_path / ".mcp.json"
        target.write_text("[1, 2]")
        with pytest.raises(McpConfigError, match="must contain a JSON object"):
            _install(target)

    def test_non_object_server_key_raises(self, tmp_path: Path) -> None:
        target = tmp_path / ".mcp.json"
        target.write_text('{"mcpServers": "nope"}')
        with pytest.raises(McpConfigError, match="must be a JSON object"):
            _install(target)

    def test_a_failed_replace_leaves_the_original_intact(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        target = tmp_path / ".mcp.json"
        original = b'{"theme":"dark"}\n'
        target.write_bytes(original)

        def fail_replace(source: object, destination: object) -> None:
            raise OSError("disk full")

        monkeypatch.setattr("agentshim.core._files.os.replace", fail_replace)
        with pytest.raises(OSError, match="disk full"):
            _install(target)

        assert target.read_bytes() == original
        assert [p.name for p in tmp_path.iterdir()] == [".mcp.json"]
