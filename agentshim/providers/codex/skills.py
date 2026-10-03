"""Codex skills: recognize a load in a shell command, and hide the user's.

Codex has no skill event in ``codex exec --json``: it injects a list of
skills into the prompt and the model loads one by reading its ``SKILL.md``
with a shell command (``sed -n '1,240p' .../.agents/skills/<name>/SKILL.md``).
That read is the only trace of the load in the stream, so it is inferred
here and nowhere else.
"""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING

from agentshim.core.events import SkillInvoked

from ._toml import toml_str

if TYPE_CHECKING:
    from collections.abc import Mapping

#: A path ending in ``skills/[<group>/...]<name>/SKILL.md``, as one shell word.
_SKILL_FILE = re.compile(
    r"(?<![^\s'\"=])(?P<path>(?:[^\s'\"]*/)?skills/(?:[^\s'\"/]+/)*?(?P<name>[^\s'\"/]+)/SKILL\.md)(?![^\s'\";|&)])"
)


def skill_reads(command: str, tool_id: str | None) -> list[SkillInvoked]:
    """Return one ``SkillInvoked`` per ``SKILL.md`` the command names, in order."""
    return [
        SkillInvoked(name=match["name"], source_path=match["path"], tool_id=tool_id)
        for match in _SKILL_FILE.finditer(command)
    ]


#: Codex's own bundled skills, which it re-extracts into every home.
_SYSTEM_SKILLS = ".system"


def user_skill_files(env: Mapping[str, str]) -> list[str]:
    """Every user-installed ``SKILL.md`` the CLI run with *env* would offer.

    Codex reads user skills from ``$CODEX_HOME/skills`` (default
    ``~/.codex/skills``) and ``~/.agents/skills``, beside the workspace's own
    ``.agents/skills``. ``$CODEX_HOME/skills/.system`` holds Codex's bundled
    skills and is not the user's. Paths are listed as found and, when a
    symlink is involved, also resolved, since either spelling may be the one
    Codex matches against. The scan runs where agentshim runs, so a CLI in a
    container is only covered when the container shares this home.
    """
    home = env.get("HOME") or os.path.expanduser("~")  # noqa: PTH111 - a str contract, not a Path
    codex_home = env.get("CODEX_HOME") or os.path.join(home, ".codex")  # noqa: PTH118
    roots = [os.path.join(codex_home, "skills"), os.path.join(home, ".agents", "skills")]  # noqa: PTH118
    found: list[str] = []
    for root in roots:
        for directory, subdirs, files in os.walk(root, followlinks=True):
            if directory == root and _SYSTEM_SKILLS in subdirs:
                subdirs.remove(_SYSTEM_SKILLS)
            subdirs.sort()
            if "SKILL.md" in files:
                path = os.path.join(directory, "SKILL.md")  # noqa: PTH118
                found.append(path)
                resolved = os.path.realpath(path)
                if resolved != path:
                    found.append(resolved)
    return list(dict.fromkeys(found))


def project_scope_overrides(env: Mapping[str, str]) -> list[str]:
    """``--config`` flags that leave Codex only the workspace's skills.

    Plugins are switched off as a feature (their skills, MCP servers and
    apps go with them), and each user skill is disabled by path through
    ``skills.config``. The override replaces any ``skills.config`` in the
    user's ``config.toml``, which is user policy this scope excludes anyway.
    """
    entries = ", ".join(
        f"{{path = {toml_str(path)}, enabled = false}}" for path in user_skill_files(env)
    )
    return [
        "--config",
        "features.plugins=false",
        "--config",
        f"skills.config=[{entries}]",
    ]
