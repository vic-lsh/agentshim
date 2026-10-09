"""Shared fixtures for the Codex app-server protocol tests."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType

REPO = Path(__file__).resolve().parents[5]
GENERATOR_PATH = REPO / "scripts" / "generate_codex_protocol.py"
SCHEMA_PATH = REPO / "scripts" / "codex_protocol" / "schema.json"
ALLOWLIST_PATH = REPO / "scripts" / "codex_protocol" / "allowlist.json"
PROTOCOL_PATH = REPO / "agentshim" / "providers" / "codex" / "app_server" / "protocol.py"
FIXTURES = REPO / "tests" / "fixtures" / "codex_app_server"


@pytest.fixture(scope="session")
def generator() -> ModuleType:
    """The generator script, imported by path (``scripts/`` is not a package)."""
    name = "generate_codex_protocol"
    spec = importlib.util.spec_from_file_location(name, GENERATOR_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolve annotations through sys.modules
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module
