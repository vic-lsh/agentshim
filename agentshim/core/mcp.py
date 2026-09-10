"""MCP server specs and the JSON config-file merge/restore used to install them.

A CONFIG_FILE provider discovers MCP servers from a JSON file in the
workspace that the user may also own. Installing merges into that file and
keeps the original bytes; restoring puts them back, and when the file changed
during the turn only the entries agentshim added are removed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

from ._files import atomic_write
from .errors import McpConfigError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path


#: Shared empty default; a frozen spec must not carry a mutable one.
_NO_ENV: Mapping[str, str] = MappingProxyType({})


@dataclass(frozen=True)
class StdioMcpServer:
    """MCP server launched as a subprocess over stdio."""

    name: str
    command: str
    args: Sequence[str] = ()
    env: Mapping[str, str] = _NO_ENV

    def __post_init__(self) -> None:
        """Reject a spec the provider could never launch.

        Validated here, at construction, because the same mistake would
        otherwise surface a turn later as an opaque provider-side MCP error.
        """
        if not self.name:
            msg = "MCP server name must not be empty"
            raise ValueError(msg)
        if not self.command:
            msg = f"MCP server {self.name!r} must declare a command"
            raise ValueError(msg)


@dataclass(frozen=True)
class HttpMcpServer:
    """MCP server reached over HTTP/SSE."""

    name: str
    url: str
    headers: Mapping[str, str] = _NO_ENV

    def __post_init__(self) -> None:
        """Reject a spec the provider could never reach.

        Validated here, at construction, because the same mistake would
        otherwise surface a turn later as an opaque provider-side MCP error.
        """
        if not self.name:
            msg = "MCP server name must not be empty"
            raise ValueError(msg)
        if not self.url:
            msg = f"MCP server {self.name!r} must declare a url"
            raise ValueError(msg)


McpServer = StdioMcpServer | HttpMcpServer


class NoopInstallation:
    """Installation for a turn that asked for no MCP servers."""

    argv: Sequence[str] = ()

    def restore(self) -> str | None:
        """Nothing was installed."""
        return None


@dataclass
class FlagsInstallation:
    """Installation for a provider that takes MCP servers as CLI flags."""

    argv: Sequence[str]

    def restore(self) -> str | None:
        """Flags live only in one argv; nothing outlives the process."""
        return None


_MISSING = object()


@dataclass(frozen=True)
class _Backup:
    original_bytes: bytes | None
    original_config: dict[str, Any]
    installed_config: dict[str, Any]
    server_key: str


class ConfigFileInstallation:
    """Undo record for one merged JSON config file.

    Owns exactly one file for the lifetime of one turn. ``restore`` is
    idempotent so a caller can run it from a ``finally`` without tracking
    whether the install succeeded.
    """

    argv: Sequence[str] = ()

    def __init__(self, target: Path, backup: _Backup) -> None:
        """Hold the undo record for one already-merged config file.

        Built by ``install_config_file`` once the merged file is on disk.
        ``backup`` describes the file as it was before the merge, which is what
        ``restore`` replays, so it must not be rebuilt from the current file.
        """
        self._target = target
        self._backup = backup
        self._done = False

    @property
    def target(self) -> Path:
        """The config file this installation merged into and will restore.

        Read-only so a caller can log or inspect the file without being able to
        point the undo record at a different one.
        """
        return self._target

    def restore(self) -> str | None:
        """Take agentshim's entries back out of the config file.

        Idempotent, so a caller can run it from a ``finally`` without tracking
        whether the install got that far. Edits the agent made to the file
        during the turn are preserved, and a file the agent deleted stays
        deleted: both are workspace changes cleanup has no right to undo.

        Never raises. The unmerge reads the file the agent may have rewritten,
        so it can find something that is no longer JSON, or an object; that is
        a normal outcome of letting an agent loose in the workspace, and a
        cleanup that raised for it would destroy the turn's result or mask the
        error the turn failed with. Anything the unmerge cannot handle falls
        back to writing the bytes the file had before the turn, and the
        returned note says so.
        """
        if self._done:
            return None
        try:
            self._unmerge()
        except (McpConfigError, OSError) as exc:
            return self._rewrite_original(exc)
        self._done = True
        return None

    def _unmerge(self) -> None:
        """Remove agentshim's entries, keeping every edit the agent made."""
        target = self._target
        backup = self._backup

        if not target.exists():
            # Deleting the config during the turn is a workspace edit, not
            # something cleanup should silently undo.
            return

        current = _load_json_object(target.read_bytes(), target)
        if current == backup.installed_config:
            if backup.original_bytes is None:
                target.unlink()
            else:
                atomic_write(target, backup.original_bytes)
            return

        # The file changed during the turn. Remove only what we added, and
        # only where the agent did not overwrite it in the meantime.
        current[backup.server_key] = _restored_servers(current, backup, target)
        if not current[backup.server_key] and backup.server_key not in backup.original_config:
            current.pop(backup.server_key, None)
        _restore_top_level_defaults(current, backup)

        if current:
            atomic_write(target, json.dumps(current, indent=2).encode())
        else:
            target.unlink()

    def _rewrite_original(self, cause: Exception) -> str:
        """Put the file back verbatim when the unmerge could not run.

        The precise unmerge needs to read the current file as JSON. When that
        is impossible the only remaining guarantee worth keeping is that the
        entries agentshim injected do not outlive the turn, so the file goes
        back to exactly the bytes it had, or is removed if it had none.
        """
        target = self._target
        original = self._backup.original_bytes
        try:
            if original is None:
                target.unlink(missing_ok=True)
            else:
                atomic_write(target, original)
        except OSError as exc:
            # Leave ``_done`` unset: a later call may still succeed, and the
            # injected entries are still in the file until one does.
            return (
                f"could not restore MCP config {target} ({cause}); "
                f"rewriting the original bytes failed too ({exc})"
            )
        self._done = True
        return (
            f"MCP config {target} could not be unmerged ({cause}); "
            "wrote back the bytes it had before the turn"
        )


def _restored_servers(current: dict[str, Any], backup: _Backup, target: Path) -> dict[str, Any]:
    original_servers = _servers_object(
        backup.original_config.get(backup.server_key, {}), backup.server_key, target
    )
    installed_servers = _servers_object(
        backup.installed_config.get(backup.server_key, {}), backup.server_key, target
    )
    restored = dict(_servers_object(current.get(backup.server_key, {}), backup.server_key, target))

    for name, installed_value in installed_servers.items():
        original_value = original_servers.get(name, _MISSING)
        if installed_value == original_value:
            continue
        if restored.get(name, _MISSING) != installed_value:
            # The agent edited this entry during the turn; leave its edit.
            continue
        if original_value is _MISSING:
            restored.pop(name, None)
        else:
            restored[name] = original_value
    return restored


def _restore_top_level_defaults(current: dict[str, Any], backup: _Backup) -> None:
    for key, installed_value in backup.installed_config.items():
        if key == backup.server_key:
            continue
        original_value = backup.original_config.get(key, _MISSING)
        if installed_value == original_value or current.get(key, _MISSING) != installed_value:
            continue
        if original_value is _MISSING:
            current.pop(key, None)
        else:
            current[key] = original_value


def install_config_file(
    target: Path,
    *,
    server_key: str,
    servers: Mapping[str, Mapping[str, Any]],
    defaults: Mapping[str, Any] | None = None,
) -> ConfigFileInstallation:
    """Merge *servers* into the JSON config at *target*, returning the undo record.

    *defaults* are top-level keys the provider needs (``$schema`` and the
    like) that are only added when the file does not already set them.
    """
    original = target.read_bytes() if target.exists() else None
    original_config = _load_json_object(original, target) if original is not None else {}
    config = dict(original_config)

    existing = _servers_object(config.get(server_key, {}), server_key, target)
    if defaults:
        for key, value in defaults.items():
            config.setdefault(key, value)
    config[server_key] = {**existing, **{name: dict(entry) for name, entry in servers.items()}}

    backup = _Backup(
        original_bytes=original,
        original_config=original_config,
        installed_config=config,
        server_key=server_key,
    )
    atomic_write(target, json.dumps(config, indent=2).encode())
    return ConfigFileInstallation(target, backup)


def _load_json_object(raw: bytes, target: Path) -> dict[str, Any]:
    try:
        loaded: object = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        msg = f"cannot merge MCP servers into invalid JSON config: {target}"
        raise McpConfigError(msg) from exc
    if not isinstance(loaded, dict):
        msg = f"MCP config must contain a JSON object: {target}"
        raise McpConfigError(msg)
    return dict(cast("dict[str, Any]", loaded))


def _servers_object(value: object, server_key: str, target: Path) -> dict[str, Any]:
    if not isinstance(value, dict):
        msg = f"{server_key!r} must be a JSON object in MCP config: {target}"
        raise McpConfigError(msg)
    return cast("dict[str, Any]", value)
