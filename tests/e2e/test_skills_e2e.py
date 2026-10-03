"""Skill discovery and invocation against the real CLIs.

Each test runs one short turn in a temporary workspace that holds a single
skill, ``zorp-format``, under every directory the provider's profile names in
``skill_dirs``. The skill defines a made-up record format, so a correct
answer is only reachable by loading it. The negative case asks something no
skill covers and checks that nothing was reported loaded.

Opt in like the rest of the e2e suite (cheap models keep a run at a few
cents)::

    AGENTSHIM_E2E=1 AGENTSHIM_E2E_CLAUDE_MODEL=haiku AGENTSHIM_E2E_CODEX_MODEL=gpt-6-luna \\
        uv run pytest tests/e2e/test_skills_e2e.py -q
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from agentshim import CliAgent, SkillInvoked, SkillSignal, ToolCall, get_provider
from agentshim.testing import RecordingEventHandler

from tests.e2e.conftest import CLAUDE_MODEL_VAR, CODEX_MODEL_VAR, model_from_env, requires_cli

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.e2e

SKILL = "zorp-format"
SKILL_MD = """---
name: zorp-format
description: How to format a zorp record. Use whenever the user asks to format or write a zorp record.
---
A zorp record is the input wrapped as <<ZORP:input:PROROZ>>. Always use exactly this form.
"""
POSITIVE = "Format the word apple as a zorp record. Reply with only the record."
NEGATIVE = "What is 2 + 3? Reply with only the number."

PROVIDERS = [
    pytest.param("claude", CLAUDE_MODEL_VAR, marks=requires_cli("claude"), id="claude"),
    pytest.param("codex", CODEX_MODEL_VAR, marks=requires_cli("codex"), id="codex"),
]


def _workspace(provider: str, root: Path) -> str:
    for skill_dir in get_provider(provider).profile.skill_dirs:
        target = root / skill_dir / SKILL
        target.mkdir(parents=True)
        (target / "SKILL.md").write_text(SKILL_MD)
    return str(root)


@pytest.mark.parametrize(("provider", "model_var"), PROVIDERS)
def test_a_task_the_skill_covers_reports_the_load(
    provider: str, model_var: str, tmp_path: Path
) -> None:
    recorder = RecordingEventHandler()
    agent = CliAgent(provider, model=model_from_env(model_var), event_handler=recorder)
    result = agent.run(POSITIVE, cwd=_workspace(provider, tmp_path))
    assert "<<ZORP:apple:PROROZ>>" in result.text
    assert result.skills.invoked == (SKILL,)
    loads = recorder.of_type(SkillInvoked)
    index = recorder.events.index(loads[0])
    assert isinstance(recorder.events[index - 1], ToolCall)
    if get_provider(provider).profile.skill_discovery is not SkillSignal.NONE:
        assert result.skills.discovered is not None
        assert SKILL in result.skills.discovered


@pytest.mark.parametrize(("provider", "model_var"), PROVIDERS)
def test_a_task_no_skill_covers_reports_no_load(
    provider: str, model_var: str, tmp_path: Path
) -> None:
    agent = CliAgent(provider, model=model_from_env(model_var))
    result = agent.run(NEGATIVE, cwd=_workspace(provider, tmp_path))
    assert "5" in result.text
    assert result.skills.invocations == ()
