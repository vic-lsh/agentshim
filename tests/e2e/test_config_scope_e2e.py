"""``ConfigScope`` against the real CLIs: the user's hooks and instructions stay out.

Each test seeds a fake user state root (the provider's relocated state root,
holding a copy of the real auth files) with a global instruction file that
names an operator code word and a ``SessionStart`` hook that touches a marker
file. The workspace holds the project's own instruction file with a project
code word. The model is asked to list every code word it was told.

``ConfigScope.ALL`` must see the operator word and run the hook, which proves
the seed is live; ``ConfigScope.PROJECT`` must see the project word and
neither the operator word nor the hook. Codex only runs a hook whose trust it
has recorded, so its turns pass ``--dangerously-bypass-hook-trust``: without
it the hook would stay silent for a reason unrelated to the scope.

Opt in like the rest of the e2e suite::

    AGENTSHIM_E2E=1 AGENTSHIM_E2E_CLAUDE_MODEL=haiku AGENTSHIM_E2E_CODEX_MODEL=gpt-6-luna \\
        uv run pytest tests/e2e/test_config_scope_e2e.py -q
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest
from agentshim import (
    CliAgent,
    ConfigScope,
    TurnRequest,
    get_provider,
    interactive_env,
    prepare_config_home,
)

from tests.e2e.conftest import CLAUDE_MODEL_VAR, CODEX_MODEL_VAR, model_from_env, requires_cli

pytestmark = pytest.mark.e2e

OPERATOR_WORD = "OPERATOR-TANGERINE"
PROJECT_WORD = "PROJECT-KIWI"
PROMPT = (
    "Do not run any tools. Reply with exactly one line listing every code word you "
    "were told about anywhere in your instructions or context (operator, project), "
    "space separated, or NONE."
)

#: Per provider: the global instruction file in its state root, the
#: workspace instruction file, the hooks file and extra turn arguments.
_LAYOUT = {
    "claude": ("CLAUDE.md", "CLAUDE.md", "settings.json", ()),
    "codex": ("AGENTS.md", "AGENTS.md", "hooks.json", ("--dangerously-bypass-hook-trust",)),
}

PROVIDERS = [
    pytest.param("claude", CLAUDE_MODEL_VAR, marks=requires_cli("claude"), id="claude"),
    pytest.param("codex", CODEX_MODEL_VAR, marks=requires_cli("codex"), id="codex"),
]


def _seeded_user_root(provider: str, root: Path, marker: Path) -> dict[str, str]:
    """A user state root with the real login, an instruction and a hook."""
    profile = get_provider(provider).profile
    assert profile.state_root_env is not None
    state_dir = profile.state_dirs[0]
    home = root / "user-state"
    home.mkdir(parents=True)
    real_home = Path(os.path.expanduser("~"))  # noqa: PTH111
    for auth_file in profile.auth_files:
        source = real_home / auth_file
        if auth_file.startswith(f"{state_dir}/") and source.is_file():
            target = home / auth_file.removeprefix(f"{state_dir}/")
            if target.name == "settings.json":
                continue  # the seeded settings below replace the user's own
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    instructions, _, hooks_file, _ = _LAYOUT[provider]
    (home / instructions).write_text(
        f"The operator code word is {OPERATOR_WORD}. Always mention it in every reply.\n"
    )
    hook = {"type": "command", "command": f"touch {marker}", "timeout": 10}
    (home / hooks_file).write_text(json.dumps({"hooks": {"SessionStart": [{"hooks": [hook]}]}}))
    if provider == "codex":
        (home / "config.toml").write_text("[features]\nhooks = true\n")
    return {profile.state_root_env: str(home)}


@pytest.mark.parametrize(("provider", "model_var"), PROVIDERS)
@pytest.mark.parametrize("scope", list(ConfigScope))
def test_project_scope_keeps_the_users_hook_and_instructions_out(
    provider: str, model_var: str, scope: ConfigScope, tmp_path: Path
) -> None:
    marker = tmp_path / "operator-hook-ran"
    env = {**interactive_env(), **_seeded_user_root(provider, tmp_path, marker)}
    if scope is ConfigScope.PROJECT:
        env.update(prepare_config_home(get_provider(provider).profile, tmp_path / "run-home", env))
    workspace = tmp_path / "ws"
    workspace.mkdir()
    _, project_file, _, extra_args = _LAYOUT[provider]
    (workspace / project_file).write_text(
        f"The project code word is {PROJECT_WORD}. Always mention it in every reply.\n"
    )

    agent = CliAgent(provider, model=model_from_env(model_var), env=env)
    result = agent.run(
        TurnRequest(PROMPT, extra_args=extra_args), cwd=str(workspace), config_scope=scope
    )

    assert PROJECT_WORD in result.text
    isolated = scope is ConfigScope.PROJECT
    assert (OPERATOR_WORD in result.text) is not isolated, result.text
    assert marker.exists() is not isolated
