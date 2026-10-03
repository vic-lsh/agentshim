"""Recognize a Codex skill load in a shell command.

Codex has no skill event in ``codex exec --json``: it injects a list of
skills into the prompt and the model loads one by reading its ``SKILL.md``
with a shell command (``sed -n '1,240p' .../.agents/skills/<name>/SKILL.md``).
That read is the only trace of the load in the stream, so it is inferred
here and nowhere else.
"""

from __future__ import annotations

import re

from agentshim.core.events import SkillInvoked

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
