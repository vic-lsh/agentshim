"""MCP server specs and the JSON config-file merge/restore."""

from __future__ import annotations

import json
import os
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
        assert server.transport == "http"

    def test_the_sse_transport_can_be_asked_for(self) -> None:
        server = HttpMcpServer(name="h", url="http://localhost:8080/sse", transport="sse")
        assert server.transport == "sse"

    def test_an_unknown_transport_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="transport must be"):
            HttpMcpServer(name="h", url="http://x", transport="websocket")  # pyright: ignore[reportArgumentType]

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
        original = (
            b'{"permissions":{"allow":["Bash(*)"]},"mcpServers":{"existing":{"command":"user"}}}\n'
        )
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
        assert json.loads(target.read_text())["mcpServers"]["vibesys-issues"] == {
            "command": "agent-edited"
        }

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


class TestSymlinkedConfig:
    """A workspace config is often a symlink into a dotfiles checkout."""

    def test_install_and_restore_follow_the_link(self, tmp_path: Path) -> None:
        real = tmp_path / "shared" / "mcp.json"
        real.parent.mkdir()
        original = b'{"mcpServers":{"existing":{"command":"user"}}}\n'
        real.write_bytes(original)
        link = tmp_path / ".mcp.json"
        link.symlink_to(real)

        installation = _install(link)

        assert link.is_symlink()
        assert json.loads(real.read_text())["mcpServers"]["vibesys-issues"]["command"] == "python"

        installation.restore()

        assert link.is_symlink()
        assert real.read_bytes() == original

    def test_the_installation_reports_the_resolved_path(self, tmp_path: Path) -> None:
        real = tmp_path / "real.json"
        real.write_text("{}")
        link = tmp_path / ".mcp.json"
        link.symlink_to(real)

        installation = _install(link)
        try:
            assert installation.target == Path(os.path.realpath(real))
        finally:
            installation.restore()

    def test_a_config_we_created_through_a_link_is_removed_from_the_real_path(
        self, tmp_path: Path
    ) -> None:
        real = tmp_path / "real.json"
        link = tmp_path / ".mcp.json"
        link.symlink_to(real)

        _install(link).restore()

        assert not real.exists()
        assert link.is_symlink()


class TestRestoreNeverRaises:
    """Restore runs from a ``finally``: raising there would destroy the turn."""

    def test_a_config_that_is_no_longer_json_falls_back_to_the_original_bytes(
        self, tmp_path: Path
    ) -> None:
        target = tmp_path / ".mcp.json"
        original = b'{"mcpServers":{"existing":{"command":"user"}}}\n'
        target.write_bytes(original)
        installation = _install(target)

        target.write_text("not json")
        note = installation.restore()

        assert target.read_bytes() == original
        assert note is not None
        assert "could not be unmerged" in note

    def test_a_config_that_is_no_longer_an_object_falls_back(self, tmp_path: Path) -> None:
        target = tmp_path / ".mcp.json"
        original = b'{"theme":"dark"}\n'
        target.write_bytes(original)
        installation = _install(target)

        target.write_text("[1, 2]")

        assert installation.restore() is not None
        assert target.read_bytes() == original

    def test_a_broken_config_we_created_is_removed(self, tmp_path: Path) -> None:
        target = tmp_path / ".mcp.json"
        installation = _install(target)

        target.write_text("not json")

        assert installation.restore() is not None
        assert not target.exists()

    def test_a_server_key_that_is_no_longer_an_object_falls_back(self, tmp_path: Path) -> None:
        target = tmp_path / ".mcp.json"
        original = b'{"theme":"dark"}\n'
        target.write_bytes(original)
        installation = _install(target)

        target.write_text('{"theme":"dark","mcpServers":"nope"}')

        assert installation.restore() is not None
        assert target.read_bytes() == original

    def test_the_fallback_is_not_repeated_on_a_second_restore(self, tmp_path: Path) -> None:
        target = tmp_path / ".mcp.json"
        original = b'{"theme":"dark"}\n'
        target.write_bytes(original)
        installation = _install(target)

        target.write_text("not json")
        assert installation.restore() is not None

        target.write_text("agent wrote this after the turn")
        assert installation.restore() is None
        assert target.read_text() == "agent wrote this after the turn"

    def test_a_restore_that_cannot_write_reports_instead_of_raising(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = tmp_path / ".mcp.json"
        target.write_bytes(b'{"theme":"dark"}\n')
        installation = _install(target)
        target.write_text("not json")

        def fail_mkstemp(*_args: object, **_kwargs: object) -> tuple[int, str]:
            message = "read-only file system"
            raise OSError(message)

        monkeypatch.setattr("agentshim.core._files.tempfile.mkstemp", fail_mkstemp)
        note = installation.restore()

        assert note is not None
        assert "read-only file system" in note

    def test_the_trivial_installations_report_nothing(self) -> None:
        assert NoopInstallation().restore() is None
        assert FlagsInstallation(["--config", "x=1"]).restore() is None


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

    def test_a_failed_replace_leaves_the_original_intact(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = tmp_path / ".mcp.json"
        original = b'{"theme":"dark"}\n'
        target.write_bytes(original)

        def fail_replace(_source: object, _destination: object) -> None:
            message = "disk full"
            raise OSError(message)

        monkeypatch.setattr("agentshim.core._files.os.replace", fail_replace)
        with pytest.raises(McpConfigError, match="disk full"):
            _install(target)

        assert target.read_bytes() == original
        assert [p.name for p in tmp_path.iterdir()] == [".mcp.json"]
