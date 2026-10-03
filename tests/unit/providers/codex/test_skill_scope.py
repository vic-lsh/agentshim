"""Codex ``SkillScope.PROJECT``: the user's skills and plugins are switched off."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from agentshim import ArgvContext, SkillScope
from agentshim.providers.codex import CodexProvider
from hypothesis import given
from hypothesis import strategies as st

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised on the 3.10 CI leg
    import tomli as tomllib  # pyright: ignore[reportMissingImports]

_NAMES = st.lists(
    st.text(alphabet="abcdefghijklmnopqrstuvwxyz-", min_size=1, max_size=12).filter(
        lambda name: name.strip("-") == name
    ),
    unique=True,
    max_size=4,
)


def _argv(env: dict[str, str], scope: SkillScope) -> list[str]:
    return CodexProvider().build_argv(
        ArgvContext(
            binary_path="/usr/local/bin/codex",
            model=None,
            env=env,
            resume_session_id=None,
            reasoning_effort=None,
            schema_inline=None,
            schema_path=None,
            skill_scope=scope,
        )
    )


def _overrides(argv: list[str]) -> dict[str, object]:
    """Parse every ``--config key=value`` the way Codex does: as TOML."""
    parsed: dict[str, object] = {}
    for index, arg in enumerate(argv[:-1]):
        if arg == "--config":
            key, _, value = argv[index + 1].partition("=")
            parsed[key] = tomllib.loads(f"v = {value}")["v"]
    return parsed


def _disabled(argv: list[str]) -> set[str]:
    entries = _overrides(argv)["skills.config"]
    assert isinstance(entries, list)
    assert all(entry["enabled"] is False for entry in entries)  # pyright: ignore[reportUnknownVariableType, reportIndexIssue]
    return {entry["path"] for entry in entries}  # pyright: ignore[reportUnknownVariableType, reportIndexIssue]


def _skill(root: Path, *parts: str) -> str:
    directory = root.joinpath(*parts)
    directory.mkdir(parents=True)
    path = directory / "SKILL.md"
    path.write_text("---\nname: x\ndescription: x\n---\n")
    return str(path)


@given(codex_home=_NAMES, agents=_NAMES, system=_NAMES)
def test_every_user_skill_is_disabled_and_no_system_skill_is(
    codex_home: list[str], agents: list[str], system: list[str]
) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        home = Path(tmp)
        user = {_skill(home, ".codex", "skills", name) for name in codex_home}
        user |= {_skill(home, ".agents", "skills", "group", name) for name in agents}
        for name in system:
            _skill(home, ".codex", "skills", ".system", name)
        argv = _argv({"HOME": tmp}, SkillScope.PROJECT)
        assert _disabled(argv) == user


def test_codex_home_relocates_the_user_skill_root(tmp_path: Path) -> None:
    moved = _skill(tmp_path, "elsewhere", "skills", "mine")
    _skill(tmp_path, ".codex", "skills", "shadowed")
    env = {"HOME": str(tmp_path), "CODEX_HOME": str(tmp_path / "elsewhere")}
    assert _disabled(_argv(env, SkillScope.PROJECT)) == {moved}


def test_a_symlinked_skill_is_disabled_under_both_spellings(tmp_path: Path) -> None:
    target = _skill(tmp_path, "store", "linked")
    skills = tmp_path / ".agents" / "skills"
    skills.mkdir(parents=True)
    (skills / "linked").symlink_to(tmp_path / "store" / "linked")
    disabled = _disabled(_argv({"HOME": str(tmp_path)}, SkillScope.PROJECT))
    assert disabled == {str(skills / "linked" / "SKILL.md"), target}


def test_project_scope_switches_plugins_off(tmp_path: Path) -> None:
    overrides = _overrides(_argv({"HOME": str(tmp_path)}, SkillScope.PROJECT))
    assert overrides["features.plugins"] is False
    assert overrides["skills.config"] == []


def test_all_scope_leaves_skills_and_plugins_alone(tmp_path: Path) -> None:
    _skill(tmp_path, ".codex", "skills", "mine")
    overrides = _overrides(_argv({"HOME": str(tmp_path)}, SkillScope.ALL))
    assert "skills.config" not in overrides
    assert "features.plugins" not in overrides
