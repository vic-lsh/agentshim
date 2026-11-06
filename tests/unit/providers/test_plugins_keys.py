"""A marketplace that cannot be used is one kind of failure, whichever way it is unusable."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from agentshim import PluginInstallError, PluginRef, SessionPlugins
from agentshim.providers.plugins import plugin_keys

if TYPE_CHECKING:
    from pathlib import Path

GOOD = {"name": "market", "plugins": [{"name": "tool"}]}


@pytest.mark.parametrize(
    "manifest",
    [
        None,
        "{",
        json.dumps([]),
        json.dumps({"plugins": []}),
        json.dumps({"name": "market"}),
        json.dumps({"name": "market", "plugins": []}),
        json.dumps({"name": "market", "plugins": [{"name": "other"}]}),
    ],
    ids=["missing", "not json", "not an object", "unnamed", "no list", "empty", "absent"],
)
def test_an_unusable_marketplace_is_a_plugin_install_error(
    tmp_path: Path, manifest: str | None
) -> None:
    (tmp_path / ".claude-plugin").mkdir()
    if manifest is not None:
        (tmp_path / ".claude-plugin" / "marketplace.json").write_text(manifest)
    plugins = SessionPlugins(tmp_path / "home", (PluginRef(tmp_path, "tool"),))
    with pytest.raises(PluginInstallError, match=r"marketplace|tool"):
        plugin_keys(plugins)


def test_a_usable_marketplace_gives_the_qualified_name(tmp_path: Path) -> None:
    (tmp_path / ".claude-plugin").mkdir()
    (tmp_path / ".claude-plugin" / "marketplace.json").write_text(json.dumps(GOOD))
    plugins = SessionPlugins(tmp_path / "home", (PluginRef(tmp_path, "tool"),))
    assert plugin_keys(plugins) == ("tool@market",)
