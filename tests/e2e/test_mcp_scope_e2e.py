"""``McpScope.SESSION`` against the real CLIs.

Two fake MCP servers answer ``whoami`` with a marker. One is seeded into a
configuration the CLI loads by default (the workspace ``.mcp.json`` for
Claude Code, a relocated ``$CODEX_HOME/config.toml`` for Codex); the other is
given to the session. The model is asked to call ``whoami`` on every server
it has and report the markers. Under ``McpScope.ALL`` both are reachable;
under ``McpScope.SESSION`` only the given one is.

Opt in like the rest of the e2e suite::

    AGENTSHIM_E2E=1 AGENTSHIM_E2E_CLAUDE_MODEL=haiku AGENTSHIM_E2E_CODEX_MODEL=gpt-6-luna \\
        uv run pytest tests/e2e/test_mcp_scope_e2e.py -q
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import pytest
from agentshim import (
    CliAgent,
    McpScope,
    StdioMcpServer,
    TurnRequest,
    get_provider,
    interactive_env,
)

from tests.e2e.conftest import CLAUDE_MODEL_VAR, CODEX_MODEL_VAR, model_from_env, requires_cli

pytestmark = pytest.mark.e2e

SERVER_SCRIPT = str(Path(__file__).with_name("fake_mcp_server.py"))
GIVEN_MARKER = "MARKER-GIVEN-7731"
SEEDED_MARKER = "MARKER-SEEDED-4192"
PROMPT = (
    "Call the `whoami` tool of every MCP server you have, one call per server. "
    "Then reply with only the exact strings they returned, one per line. "
    "If you have no such tool, reply with only NONE."
)


def _given() -> StdioMcpServer:
    return StdioMcpServer("given", sys.executable, args=(SERVER_SCRIPT, GIVEN_MARKER))


def _seeded_args() -> list[str]:
    return [SERVER_SCRIPT, SEEDED_MARKER]


@requires_cli("claude")
@pytest.mark.parametrize("scope", list(McpScope))
def test_claude_session_scope_hides_the_projects_own_servers(
    scope: McpScope, tmp_path: Path
) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    config = {"mcpServers": {"seeded": {"command": sys.executable, "args": _seeded_args()}}}
    (workspace / ".mcp.json").write_text(json.dumps(config))
    agent = CliAgent("claude", model=model_from_env(CLAUDE_MODEL_VAR), env=interactive_env())
    session = agent.start_session(cwd=str(workspace), mcp_scope=scope)
    result = session.turn(_request())
    assert GIVEN_MARKER in result.text
    assert (SEEDED_MARKER in result.text) is (scope is McpScope.ALL)


def _request() -> TurnRequest:
    return TurnRequest(PROMPT, mcp_servers=[_given()])


def _codex_home_with_a_server(root: Path) -> dict[str, str]:
    """A relocated ``$CODEX_HOME``: the real auth files plus one seeded server."""
    profile = get_provider("codex").profile
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
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.toml").write_text(
        "[mcp_servers.seeded]\n"
        f"command = {json.dumps(sys.executable)}\n"
        f"args = {json.dumps(_seeded_args())}\n"
    )
    return {profile.state_root_env: str(home)}


@requires_cli("codex")
@pytest.mark.parametrize("scope", list(McpScope))
def test_codex_session_scope_hides_the_users_own_servers(scope: McpScope, tmp_path: Path) -> None:
    env = {**interactive_env(), **_codex_home_with_a_server(tmp_path)}
    workspace = tmp_path / "ws"
    workspace.mkdir()
    agent = CliAgent("codex", model=model_from_env(CODEX_MODEL_VAR), env=env)
    session = agent.start_session(cwd=str(workspace), mcp_scope=scope)
    result = session.turn(_request())
    assert GIVEN_MARKER in result.text
    assert (SEEDED_MARKER in result.text) is (scope is McpScope.ALL)
