"""Explicit plugin activation over Claude's project settings merge."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

from agentshim.core.stream import parse_json_object
from agentshim.providers.plugins import plugin_keys

if TYPE_CHECKING:
    from agentshim.core.plugins import SessionPlugins


def plugin_settings(plugins: SessionPlugins, cwd: str | None) -> dict[str, bool]:
    """Disable project plugin selections before enabling the session's own.

    Claude merges ``enabledPlugins`` by key, so an inline map containing only
    the selected plugins would leave project-enabled plugins active. The fresh
    registry excludes the operator's plugins; these explicit false entries
    exclude selections inherited from workspace settings as well.
    """
    enabled: dict[str, bool] = {}
    if cwd is not None:
        directory = Path(cwd).absolute()
        for ancestor in (directory, *directory.parents):
            for filename in ("settings.json", "settings.local.json"):
                path = ancestor / ".claude" / filename
                if path.is_file():
                    data = parse_json_object(path.read_text(encoding="utf-8"))
                    entries: object = None if data is None else data.get("enabledPlugins")
                    if isinstance(entries, dict):
                        enabled.update(dict.fromkeys(cast("dict[str, object]", entries), False))
    enabled.update(dict.fromkeys(plugin_keys(plugins), True))
    return enabled
