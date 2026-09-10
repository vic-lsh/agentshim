"""The public surface is what ``__all__`` says it is."""

from __future__ import annotations

import ast
from pathlib import Path

import agentshim
import agentshim.testing


def test_every_exported_name_is_importable() -> None:
    for name in agentshim.__all__:
        assert hasattr(agentshim, name), name


def test_exported_names_are_sorted_and_unique() -> None:
    assert agentshim.__all__ == sorted(agentshim.__all__)
    assert len(agentshim.__all__) == len(set(agentshim.__all__))


def test_testing_module_exports() -> None:
    for name in agentshim.testing.__all__:
        assert hasattr(agentshim.testing, name), name


def test_version_is_reported() -> None:
    assert agentshim.__version__ == "0.6.0"


def test_core_and_execution_do_not_import_providers_at_module_level() -> None:
    """The layering rule, checked instead of documented.

    ``import-linter`` enforces the whole ordering as a build gate; this
    catches the specific edge that used to exist in ``core.agent`` without
    needing the tool installed.
    """
    root = Path(agentshim.__file__ or "").parent
    offenders: list[str] = []
    for path in sorted((*(root / "core").rglob("*.py"), *(root / "execution").rglob("*.py"))):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            for statement in (
                ast.walk(node) if isinstance(node, (ast.Import, ast.ImportFrom)) else ()
            ):
                if isinstance(statement, ast.ImportFrom) and "providers" in (
                    statement.module or ""
                ):
                    offenders.append(f"{path.name}: {ast.unparse(statement)}")
                if isinstance(statement, ast.Import):
                    offenders.extend(
                        f"{path.name}: import {alias.name}"
                        for alias in statement.names
                        if "providers" in alias.name
                    )
    assert offenders == []


def test_provider_names_lists_the_ported_providers() -> None:
    assert agentshim.provider_names() == ["claude", "codex", "copilot", "gemini", "opencode"]
