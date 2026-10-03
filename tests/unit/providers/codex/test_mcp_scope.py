"""Codex ``McpScope.SESSION``: every MCP server the session was not given is off."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from agentshim import ArgvContext, McpScope, SkillScope, StdioMcpServer
from agentshim.providers.codex import CodexProvider, parse_mcp_servers
from hypothesis import given
from hypothesis import strategies as st

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised on the 3.10 CI leg
    import tomli as tomllib  # pyright: ignore[reportMissingImports]

_KEYS = st.lists(
    st.text(alphabet="abcdefghijklmnopqrstuvwxyz_", min_size=1, max_size=10),
    unique=True,
    max_size=5,
)


def _argv(
    env: dict[str, str],
    *,
    mcp_scope: McpScope = McpScope.SESSION,
    servers: tuple[StdioMcpServer, ...] = (),
    cwd: str | None = None,
    skill_scope: SkillScope = SkillScope.ALL,
) -> list[str]:
    return CodexProvider().build_argv(
        ArgvContext(
            binary_path="/usr/local/bin/codex",
            model=None,
            env=env,
            resume_session_id=None,
            reasoning_effort=None,
            schema_inline=None,
            schema_path=None,
            cwd=cwd,
            skill_scope=skill_scope,
            mcp_scope=mcp_scope,
            mcp_servers=servers,
            mcp_argv=CodexProvider().install_mcp(None, servers).argv,
        )
    )


def _overrides(argv: list[str]) -> dict[str, object]:
    parsed: dict[str, object] = {}
    for index, arg in enumerate(argv[:-1]):
        if arg == "--config":
            key, _, value = argv[index + 1].partition("=")
            parsed[key] = tomllib.loads(f"v = {value}")["v"]
    return parsed


def _disabled_servers(argv: list[str]) -> set[str]:
    return {
        key.removeprefix("mcp_servers.").removesuffix(".enabled")
        for key, value in _overrides(argv).items()
        if key.startswith("mcp_servers.") and key.endswith(".enabled") and value is False
    }


def _write_config(path: Path, keys: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f'[mcp_servers.{key}]\ncommand = "x"\n' for key in keys))


@given(user=_KEYS, project=_KEYS)
def test_every_configured_server_is_disabled(user: list[str], project: list[str]) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        home = Path(tmp) / "home"
        workspace = Path(tmp) / "ws"
        _write_config(home / ".codex" / "config.toml", user)
        _write_config(workspace / ".codex" / "config.toml", project)
        argv = _argv({"HOME": str(home)}, cwd=str(workspace))
        assert _disabled_servers(argv) == set(user) | set(project)


@given(user=_KEYS, given_names=_KEYS)
def test_a_given_server_is_never_disabled(user: list[str], given_names: list[str]) -> None:
    servers = tuple(StdioMcpServer(name.replace("_", "-"), "tool") for name in given_names)
    with tempfile.TemporaryDirectory() as tmp:
        _write_config(Path(tmp) / ".codex" / "config.toml", user)
        argv = _argv({"HOME": tmp}, servers=servers)
        assert _disabled_servers(argv) == set(user) - set(given_names)
        assert set(parse_mcp_servers(argv)) == set(given_names)


def test_plugins_and_apps_are_switched_off(tmp_path: Path) -> None:
    overrides = _overrides(_argv({"HOME": str(tmp_path)}))
    assert overrides["features.plugins"] is False
    assert overrides["features.apps"] is False


def test_a_server_name_that_is_not_a_bare_key_is_quoted(tmp_path: Path) -> None:
    config = tmp_path / ".codex" / "config.toml"
    config.parent.mkdir()
    config.write_text('[mcp_servers."dotted.name"]\ncommand = "x"\n')
    argv = _argv({"HOME": str(tmp_path)})
    assert _disabled_servers(argv) == {'"dotted.name"'}


def test_codex_home_relocates_the_user_config(tmp_path: Path) -> None:
    _write_config(tmp_path / "elsewhere" / "config.toml", ["moved"])
    _write_config(tmp_path / ".codex" / "config.toml", ["shadowed"])
    env = {"HOME": str(tmp_path), "CODEX_HOME": str(tmp_path / "elsewhere")}
    assert _disabled_servers(_argv(env)) == {"moved"}


def test_all_scope_leaves_servers_alone(tmp_path: Path) -> None:
    _write_config(tmp_path / ".codex" / "config.toml", ["mine"])
    overrides = _overrides(_argv({"HOME": str(tmp_path)}, mcp_scope=McpScope.ALL))
    assert not [key for key in overrides if key.startswith("mcp_servers.")]
    assert "features.apps" not in overrides


def test_both_scopes_together_emit_the_plugins_flag_once(tmp_path: Path) -> None:
    argv = _argv({"HOME": str(tmp_path)}, skill_scope=SkillScope.PROJECT)
    assert argv.count("features.plugins=false") == 1


@given(user=_KEYS)
def test_every_disabled_server_restates_its_transport(user: list[str]) -> None:
    """Disabling alone fails a CLI that cannot see the file ("invalid transport").

    The turn's argv must therefore define each disabled entry completely, so
    it is valid whether or not Codex reads the config file agentshim scanned.
    """
    with tempfile.TemporaryDirectory() as tmp:
        _write_config(Path(tmp) / ".codex" / "config.toml", user)
        overrides = _overrides(_argv({"HOME": tmp}))
        for key in user:
            assert overrides[f"mcp_servers.{key}.command"] == "x"
            assert overrides[f"mcp_servers.{key}.enabled"] is False


def test_a_remote_server_restates_its_url(tmp_path: Path) -> None:
    config = tmp_path / ".codex" / "config.toml"
    config.parent.mkdir()
    config.write_text('[mcp_servers.remote]\nurl = "https://example.invalid/mcp"\n')
    overrides = _overrides(_argv({"HOME": str(tmp_path)}))
    assert overrides["mcp_servers.remote.url"] == "https://example.invalid/mcp"
    assert "mcp_servers.remote.command" not in overrides
