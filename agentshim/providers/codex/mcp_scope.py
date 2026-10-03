"""Codex MCP servers: hide every one the session was not given.

Codex merges MCP servers from several layers: ``$CODEX_HOME/config.toml``,
system config, a trusted workspace's ``.codex/config.toml``, installed
plugins, and the account's ChatGPT apps (the ``codex_apps`` server). A
``--config mcp_servers={}`` override does not replace the merged table, so
each configured server is switched off by name (``enabled = false``), and
plugins and apps are switched off as features.
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING, Any, cast

from ._toml import toml_str

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised on the 3.10 CI leg
    import tomli as tomllib  # pyright: ignore[reportMissingImports]

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable, Mapping

_SYSTEM_CONFIG = "/etc/codex/config.toml"
_BARE_KEY = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-")

#: Features that bring in MCP servers on their own: plugins (each may bundle
#: servers) and apps (the account's ChatGPT connectors).
_MCP_FEATURES = ("plugins", "apps")


def configured_server_keys(env: Mapping[str, str], cwd: str | None) -> list[str]:
    """Every ``mcp_servers`` key the config files Codex would read define.

    Reads the user's ``$CODEX_HOME/config.toml`` (default ``~/.codex``), the
    system config, and ``<cwd>/.codex/config.toml``. A file that is missing or
    not valid TOML contributes nothing (Codex reports a broken config itself).
    The scan runs where agentshim runs, so a CLI in a container is only
    covered when the container shares these files.
    """
    home = env.get("HOME") or os.path.expanduser("~")  # noqa: PTH111 - a str contract, not a Path
    codex_home = env.get("CODEX_HOME") or os.path.join(home, ".codex")  # noqa: PTH118
    paths = [os.path.join(codex_home, "config.toml"), _SYSTEM_CONFIG]  # noqa: PTH118
    if cwd is not None:
        paths.append(os.path.join(cwd, ".codex", "config.toml"))  # noqa: PTH118
    keys: list[str] = []
    for path in paths:
        keys += _server_keys(path)
    return list(dict.fromkeys(keys))


def _server_keys(path: str) -> Iterable[str]:
    config = _read_toml(path)
    servers: object = config.get("mcp_servers")
    if not isinstance(servers, dict):
        return []
    return [str(key) for key in cast("dict[object, object]", servers)]


def _read_toml(path: str) -> dict[str, Any]:
    try:
        with open(path, "rb") as handle:  # noqa: PTH123 - a str contract, not a Path
            # pyright analyzes the 3.10 leg, where the optional ``tomli`` has no
            # stubs in the 3.12 environment: the types are known at runtime.
            return tomllib.load(handle)  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
    except (OSError, tomllib.TOMLDecodeError):  # pyright: ignore[reportUnknownMemberType]
        return {}


def _table_key(key: str) -> str:
    return key if key and set(key) <= _BARE_KEY else toml_str(key)


def session_scope_overrides(
    env: Mapping[str, str], cwd: str | None, given_keys: Collection[str]
) -> list[str]:
    """``--config`` flags that leave Codex only the servers in *given_keys*.

    *given_keys* are the ``mcp_servers`` keys the turn's own overrides define;
    they are not disabled. A given server that shares a key with a configured
    one is merged field by field with it by Codex, so give session servers
    names the user's config does not use.
    """
    flags: list[str] = []
    for key in configured_server_keys(env, cwd):
        if key not in given_keys:
            flags += ["--config", f"mcp_servers.{_table_key(key)}.enabled=false"]
    for feature in _MCP_FEATURES:
        flags += ["--config", f"features.{feature}=false"]
    return flags
