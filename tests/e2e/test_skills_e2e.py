"""Skill discovery and invocation against the real CLIs.

Each test runs one short turn in a temporary workspace that holds a single
skill, ``zorp-format``, under every directory the provider's profile names in
``skill_dirs``. The skill defines a made-up record format, so a correct
answer is only reachable by loading it. The negative case asks something no
skill covers and checks that nothing was reported loaded.

The scope tests also seed a fake user home (the provider's relocated state
root, holding a copy of the real auth files) with a second skill, and check
that ``SkillScope.PROJECT`` offers the workspace skill and not the home one,
while ``SkillScope.ALL`` offers both.

Opt in like the rest of the e2e suite (cheap models keep a run at a few
cents)::

    AGENTSHIM_E2E=1 AGENTSHIM_E2E_CLAUDE_MODEL=haiku AGENTSHIM_E2E_CODEX_MODEL=gpt-6-luna \\
        uv run pytest tests/e2e/test_skills_e2e.py -q
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from agentshim import (
    ArgvContext,
    CliAgent,
    SkillInvoked,
    SkillScope,
    SkillSignal,
    ToolCall,
    get_provider,
    interactive_env,
)
from agentshim.testing import RecordingEventHandler

from tests.e2e.conftest import CLAUDE_MODEL_VAR, CODEX_MODEL_VAR, model_from_env, requires_cli

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


HOME_SKILL = "home-only-skill"


def _home_with_a_skill(provider: str, root: Path) -> dict[str, str]:
    """A relocated user state root: the real auth files plus one user skill.

    Returns the environment that points the CLI at it.
    """
    profile = get_provider(provider).profile
    assert profile.state_root_env is not None
    state_dir = profile.state_dirs[0]
    home = root / "user-state"
    real_home = Path(os.path.expanduser("~"))  # noqa: PTH111
    for auth_file in profile.auth_files:
        source = real_home / auth_file
        if auth_file.startswith(f"{state_dir}/") and source.is_file():
            target = home / auth_file.removeprefix(f"{state_dir}/")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    skill = home / "skills" / HOME_SKILL
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(SKILL_MD.replace(SKILL, HOME_SKILL))
    return {profile.state_root_env: str(home)}


@requires_cli("claude")
@pytest.mark.parametrize("scope", list(SkillScope))
def test_claude_project_scope_offers_only_the_workspace_skills(
    scope: SkillScope, tmp_path: Path
) -> None:
    env = {**interactive_env(), **_home_with_a_skill("claude", tmp_path)}
    agent = CliAgent("claude", model=model_from_env(CLAUDE_MODEL_VAR), env=env)
    workspace = _workspace("claude", tmp_path / "ws")
    result = agent.run(NEGATIVE, cwd=workspace, skill_scope=scope)
    assert "5" in result.text
    offered = result.skills.discovered
    assert offered is not None
    assert SKILL in offered
    assert (HOME_SKILL in offered) is (scope is SkillScope.ALL)


def _codex_offered(env: dict[str, str], workspace: str, scope: SkillScope) -> list[str]:
    """The skill names Codex puts in the model's prompt, without a model call.

    ``codex debug prompt-input`` renders the prompt ``codex exec`` would send,
    under the same ``--config`` overrides, which are taken from the argv the
    provider builds for *scope*.
    """
    argv = get_provider("codex").build_argv(
        ArgvContext(
            binary_path="codex",
            model=None,
            env=env,
            resume_session_id=None,
            reasoning_effort=None,
            schema_inline=None,
            schema_path=None,
            skill_scope=scope,
        )
    )
    overrides: list[str] = []
    for index, arg in enumerate(argv[:-1]):
        if arg == "--config":
            overrides += ["--config", argv[index + 1]]
    completed = subprocess.run(  # noqa: S603
        [shutil.which("codex") or "codex", "debug", "prompt-input", *overrides, "hi"],
        capture_output=True,
        text=True,
        check=True,
        cwd=workspace,
        env=env,
        timeout=120,
    )
    prompt = json.dumps(json.loads(completed.stdout))
    return re.findall(r"\\n- ([\w:.-]+): ", prompt)


@requires_cli("codex")
@pytest.mark.parametrize("scope", list(SkillScope))
def test_codex_project_scope_offers_only_the_workspace_skills(
    scope: SkillScope, tmp_path: Path
) -> None:
    env = {**interactive_env(), **_home_with_a_skill("codex", tmp_path)}
    offered = _codex_offered(env, _workspace("codex", tmp_path / "ws"), scope)
    assert SKILL in offered
    assert (HOME_SKILL in offered) is (scope is SkillScope.ALL)
