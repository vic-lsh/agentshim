"""Install a session's explicit plugins without reading an operator's home.

The native installers own marketplace semantics, including ``strict: false``
entries with inline component paths. This module owns only the fresh home and
its binding to one immutable selection. Callers retain that home for resume.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from typing import TYPE_CHECKING, cast

from agentshim.core.errors import PluginInstallError, ProviderCapabilityError
from agentshim.core.home import prepare_config_home
from agentshim.core.stream import parse_json_object
from agentshim.execution.executor import CommandRequest, NullSink

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from agentshim.core.plugins import SessionPlugins
    from agentshim.core.profile import ProviderProfile
    from agentshim.execution.executor import CommandExecutor

_MARKER = ".agentshim-plugins.json"
_INSTALL_TIMEOUT = 60.0


def plugin_keys(plugins: SessionPlugins) -> tuple[str, ...]:
    """Validate local marketplace selectors and return native qualified names."""
    keys: list[str] = []
    roots: dict[str, Path] = {}
    for ref in plugins.refs:
        manifest = ref.marketplace / ".claude-plugin" / "marketplace.json"
        data = parse_json_object(manifest.read_text(encoding="utf-8"))
        name = None if data is None else data.get("name")
        entries = None if data is None else data.get("plugins")
        if (
            not isinstance(name, str)
            or not name
            or not name[0].isalnum()
            or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in name)
            or not isinstance(entries, list)
        ):
            msg = f"invalid plugin marketplace manifest: {manifest}"
            raise ValueError(msg)
        if not any(_names_plugin(p, ref.name) for p in cast("list[object]", entries)):
            msg = f"plugin {ref.name!r} is absent from {manifest}"
            raise ValueError(msg)
        if name in roots and roots[name] != ref.marketplace:
            msg = f"marketplace name {name!r} refers to multiple checkouts"
            raise ValueError(msg)
        roots[name] = ref.marketplace
        key = f"{ref.name}@{name}"
        if key in keys:
            msg = f"duplicate session plugin: {key}"
            raise ValueError(msg)
        keys.append(key)
    return tuple(keys)


def prepare_plugins(
    profile: ProviderProfile,
    plugins: SessionPlugins,
    env: Mapping[str, str],
    executor: CommandExecutor,
    binary: str,
) -> dict[str, str]:
    """Return an isolated launch environment, installing once in a fresh home.

    An existing home must bear this library's completed installation record for
    exactly this selection and provider. Failed installations remove the home;
    a pre-existing unowned directory is never modified. Installer failures are
    errors, never an invitation to run without the requested plugins.
    """
    keys = plugin_keys(plugins)
    record = json.dumps(
        {"provider": profile.name, "refs": [[str(r.marketplace), r.name] for r in plugins.refs]},
        sort_keys=True,
    )
    home = plugins.home
    marker = home / _MARKER
    if home.exists():
        if not marker.is_file() or marker.read_text(encoding="utf-8") != record:
            msg = "plugin home must be fresh or belong to this exact session plugin selection"
            raise PluginInstallError(msg)
    else:
        home.mkdir(parents=True)
        try:
            isolated = _isolated_env(profile, home, env)
            registered: set[Path] = set()
            for ref, key in zip(plugins.refs, keys, strict=True):
                commands = (
                    ["plugin", "marketplace", "add", str(ref.marketplace)],
                    [*profile.plugin_install_command, key],
                )
                for command in commands:
                    if command[1] == "marketplace" and ref.marketplace in registered:
                        continue
                    _install(
                        executor,
                        CommandRequest(
                            argv=[binary, *command],
                            stdin=None,
                            cwd=str(home),
                            env=isolated,
                            timeout=_INSTALL_TIMEOUT,
                        ),
                    )
                registered.add(ref.marketplace)
            marker.write_text(record, encoding="utf-8")
        except BaseException:
            shutil.rmtree(home)
            raise
    return _isolated_env(profile, home, env)


def _isolated_env(profile: ProviderProfile, home: Path, env: Mapping[str, str]) -> dict[str, str]:
    # Claude isolates ordinary config by flags, but plugin installation needs a
    # separate registry. Reuse the credential-link mechanism for that home too.
    files = tuple(
        name.removeprefix(profile.state_dirs[0] + "/") for name in profile.credential_files
    )
    isolated = dict(env)
    isolated.update(prepare_config_home(replace(profile, config_home_files=files), home, env))
    return isolated


def _install(executor: CommandExecutor, request: CommandRequest) -> None:
    result = executor.run(request, NullSink())
    if result.returncode:
        msg = f"plugin installation failed: {result.stderr or result.stdout}"
        raise PluginInstallError(msg)


def _names_plugin(entry: object, name: str) -> bool:
    return isinstance(entry, dict) and cast("dict[str, object]", entry).get("name") == name


def check_plugin_mcp(plugins: SessionPlugins) -> None:
    """Refuse plugin-bundled servers when only session MCP servers are allowed.

    Codex has no switch to disable just a plugin's MCP component. Local,
    skill-only plugins are compatible; an uninspectable source is refused.
    """
    from_data = [
        (
            ref,
            parse_json_object(
                (ref.marketplace / ".claude-plugin" / "marketplace.json").read_text(
                    encoding="utf-8"
                )
            ),
        )
        for ref in plugins.refs
    ]
    for ref, data in from_data:
        entries: object = () if data is None else data.get("plugins", ())
        for entry in cast("list[object]", entries):
            if not _names_plugin(entry, ref.name):
                continue
            fields = cast("dict[str, object]", entry)
            source = fields.get("source")
            if not isinstance(source, str) or not source.startswith("./"):
                _mcp_incompatible(ref.name)
            directory = ref.marketplace / cast("str", source)
            manifests = [
                directory / ".claude-plugin/plugin.json",
                directory / ".codex-plugin/plugin.json",
            ]
            if fields.get("mcpServers") or (directory / ".mcp.json").exists():
                _mcp_incompatible(ref.name)
            for manifest in manifests:
                if manifest.is_file():
                    content = parse_json_object(manifest.read_text(encoding="utf-8"))
                    if content is None or content.get("mcpServers"):
                        _mcp_incompatible(ref.name)


def _mcp_incompatible(name: str) -> None:
    msg = f"plugin {name!r} cannot guarantee McpScope.SESSION: only inspectable local plugins without MCP servers are supported"
    raise ProviderCapabilityError(msg)
