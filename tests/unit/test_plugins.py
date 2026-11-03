from pathlib import Path

import pytest
from agentshim import CliAgent, PluginRef, ProviderCapabilityError, SessionPlugins
from agentshim.testing import FakeExecutor


def test_plugin_paths_and_names_are_unambiguous(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="absolute"):
        PluginRef(Path("relative"), "example")
    with pytest.raises(ValueError, match="invalid plugin name"):
        PluginRef(tmp_path, "example@marketplace")
    with pytest.raises(ValueError, match="absolute"):
        SessionPlugins(Path("relative"))
    assert PluginRef(tmp_path, "example").marketplace == tmp_path


def test_unsupported_provider_refuses_even_an_empty_selection(tmp_path: Path) -> None:
    agent = CliAgent("gemini", executor=FakeExecutor([]))
    with pytest.raises(ProviderCapabilityError, match="session plugins"):
        agent.start_session(plugins=SessionPlugins(tmp_path / "session"))
